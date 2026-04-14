#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#ifdef USE_CHIPYARD
#include <riscv-pk/encoding.h>
#endif

#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#include "nn.h"
#include "profile.h"
#include "mico_nn.h"
#include "mico_quant.h"

#include "llama2_config.h"

#ifndef CONTEXT_LEN
#define CONTEXT_LEN 128
#endif

#ifndef LLAMA_LAYERS
#define LLAMA_LAYERS 6
#endif


#ifdef QUANTIZED
typedef Tensor2D_Q8 DataType;
typedef qbyte WeightType;
#else
typedef Tensor2D_F32 DataType;
typedef float WeightType;
#endif

#ifdef USE_INT8_KV
typedef int8_t kv_type;
#else
typedef float kv_type;
#endif

#ifndef LLAMA2_BIN
#define LLAMA2_BIN "./llama2/llama_model.bin"
#endif

//#define TEST_DOTP

INCLUDE_FILE(".rodata", LLAMA2_BIN, llama_model);
extern uint8_t llama_model_data[];
extern size_t llama_model_start[];
extern size_t llama_model_end[];

INCLUDE_FILE(".rodata", "./llama2/tokenizer.bin", tokenizer);
extern uint8_t tokenizer_data[];
extern size_t tokenizer_start[];
extern size_t tokenizer_end[];
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    DataType* wq; // (layer, dim, n_heads * head_size)
    DataType* wk; // (layer, dim, n_kv_heads * head_size)
    DataType* wv; // (layer, dim, n_kv_heads * head_size)
    DataType* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    DataType* w1; // (layer, hidden_dim, dim)
    DataType* w2; // (layer, dim, hidden_dim)
    DataType* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    DataType wcls;
} TransformerWeights;

typedef struct{
    qtype* wq_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wk_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wv_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* wo_qtype; // (layer * 2, i - wq, i + 1 - aq)
    // weights for ffn
    qtype* w1_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* w2_qtype; // (layer * 2, i - wq, i + 1 - aq)
    qtype* w3_qtype; // (layer * 2, i - wq, i + 1 - aq)

} TransformerQScheme;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    kv_type* key_cache; // (layer, seq_len, dim)
    kv_type* value_cache; // (layer, seq_len, dim)
    #ifdef USE_INT8_KV
    float* key_scales; // (layer, seq_len)
    float* value_scales; // (layer, seq_len)
    #endif
    // RoPE caches
    float* rope_inv_freq; // (head_size/2)
    float* rope_cos; // (seq_len, head_size/2)
    float* rope_sin; // (seq_len, head_size/2)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    TransformerQScheme qscheme; // quantization scheme for the weights
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return MiCo_time();
}

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    printf("Start Allocating Buffers\n");
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    int head_pairs = head_size / 2;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    #ifdef USE_INT8_KV
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    size_t kv_scale_elems = (size_t)p->n_layers * p->seq_len;
    size_t kv_cache_elems = (size_t)p->n_layers * p->seq_len * kv_dim;
    s->key_scales = calloc(kv_scale_elems, sizeof(float));
    s->value_scales = calloc(kv_scale_elems, sizeof(float));
    printf("Allocate KV Quant Buffer + Scales of size %ld Bytes...\n",
        (long)((kv_dim * 2 + kv_scale_elems * 2) * sizeof(float)));
    s->key_cache = calloc(kv_cache_elems, sizeof(kv_type));
    s->value_cache = calloc(kv_cache_elems, sizeof(kv_type));
    printf("Allocate KV Cache of size %ld KB...\n",
        (long)((kv_cache_elems * sizeof(kv_type)) / 1024));
    #else
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kv_type));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kv_type));
    printf("Allocate KV Cache of size %ld KB...\n",
        (long)((p->n_layers * p->seq_len * kv_dim * sizeof(kv_type)) / 1024));
    #endif

    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // RoPE precompute: inv_freq per head (shared across heads)
    s->rope_inv_freq = calloc(1, sizeof(float));
    // Precompute cos/sin for all positions and per-head pairs
    // size_t tbl_elems = (size_t)p->seq_len * head_pairs;
    s->rope_cos = calloc(1, sizeof(float));
    s->rope_sin = calloc(1, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->rope_inv_freq || !s->rope_cos || !s->rope_sin) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    long end = MiCo_time();
    long total_run_state_size = p->dim * sizeof(float) * 4;
    total_run_state_size += p->hidden_dim * sizeof(float) * 2;
    total_run_state_size += p->n_layers * p->seq_len * kv_dim * sizeof(kv_type) * 2;
    total_run_state_size += p->n_heads * p->seq_len * sizeof(float);
    total_run_state_size += p->vocab_size * sizeof(float);
    total_run_state_size += head_pairs * sizeof(float) * (1 + 2 * p->seq_len);
    printf("Total Run State Size: %ld KB\n", total_run_state_size / 1024);
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
#ifdef USE_INT8_KV
    free(s->key_scales);
    free(s->value_scales);
    free(s->key_cache);
    free(s->value_cache);
#else
    free(s->key_cache);
    free(s->value_cache);
#endif
}

size_t init_weight(DataType* w, char* ptr, int n_layers, int n, int m){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].shape[0] = n;
        w[i].shape[1] = m;
        w[i].data = (WeightType*) ptr;
        ptr += n * m * sizeof(WeightType);
    }
    return ptr - ptr0;
}

size_t init_quant_weight(DataType* w, char* ptr, int n_layers, int n, int m){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].shape[0] = n;
        w[i].shape[1] = m;
        w[i].data = (WeightType*) ptr;
        ptr += n * m * sizeof(WeightType) / (8 / w[i].wq);
    }
    return ptr - ptr0;
}

size_t init_quant_weight_scale(DataType* w, char* ptr, int n_layers){
    char* ptr0 = ptr;
    for (int i = 0; i < n_layers; i++) {
        w[i].scale = *(float*)ptr;
        ptr += sizeof(float);
    }
    return ptr - ptr0;
}

void* init_float_params(
    TransformerWeights *w, 
    Config* p, 
    char* ptr, int shared_weights){

    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    w->rms_att_weight = (float*) ptr;
    ptr += n_layers * p->dim * sizeof(float);

    w->rms_ffn_weight = (float*) ptr;
    ptr += n_layers * p->dim * sizeof(float);

    w->rms_final_weight = (float*) ptr;
    ptr += p->dim * sizeof(float);

    return ptr;
}

void* init_qschemes(
    TransformerQScheme *wq,
    Config* p,
    char* ptr
){
    int n_layers = p->n_layers;

    wq->wq_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wk_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wv_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->wo_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w1_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w2_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    wq->w3_qtype = (qtype*)ptr;
    ptr += 2 * n_layers * sizeof(qtype);

    return ptr;
}

void init_weight_qtypes(qtype* wq, DataType*w, int n_layers){
    for (int i = 0; i < n_layers; i++) {
        w[i].wq = wq[2*i];
    }
}

void memory_map_weights(
    TransformerWeights *w, 
    TransformerQScheme *q,
    Config* p, 
    char* ptr, int shared_weights){
    
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    size_t inc = 0;
    w->wq = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wq_qtype, w->wq, n_layers);
    ptr += init_quant_weight(w->wq, ptr, n_layers, p->dim, p->n_heads * head_size);
    #else
    ptr += init_weight(w->wq, ptr, n_layers, p->dim, p->n_heads * head_size);
    #endif

    w->wk = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wk_qtype, w->wk, n_layers);
    ptr += init_quant_weight(w->wk, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #else
    ptr += init_weight(w->wk, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #endif

    w->wv = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wv_qtype, w->wv, n_layers);
    ptr += init_quant_weight(w->wv, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #else
    ptr += init_weight(w->wv, ptr, n_layers, p->dim, p->n_kv_heads * head_size);
    #endif

    w->wo = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->wo_qtype, w->wo, n_layers);
    ptr += init_quant_weight(w->wo, ptr, n_layers, p->n_heads * head_size, p->dim);
    #else
    ptr += init_weight(w->wo, ptr, n_layers, p->n_heads * head_size, p->dim);
    #endif

    w->w1 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w1_qtype, w->w1, n_layers);
    ptr += init_quant_weight(w->w1, ptr, n_layers, p->hidden_dim, p->dim);
    #else
    ptr += init_weight(w->w1, ptr, n_layers, p->hidden_dim, p->dim);
    #endif

    w->w2 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w2_qtype, w->w2, n_layers);
    ptr += init_quant_weight(w->w2, ptr, n_layers, p->dim, p->hidden_dim);
    #else
    ptr += init_weight(w->w2, ptr, n_layers, p->dim, p->hidden_dim);
    #endif

    w->w3 = (DataType*)malloc(n_layers * sizeof(DataType));
    #ifdef QUANTIZED
    init_weight_qtypes(q->w3_qtype, w->w3, n_layers);
    ptr += init_quant_weight(w->w3, ptr, n_layers, p->hidden_dim, p->dim);
    #else
    ptr += init_weight(w->w3, ptr, n_layers, p->hidden_dim, p->dim);
    #endif
    if(shared_weights){
        #ifdef QUANTIZED
        // Use INT8 for the final classifier to save memory
        w->wcls.shape[0] = p->vocab_size;
        w->wcls.shape[1] = p->dim;
        w->wcls.data = (WeightType*) ptr;
        w->wcls.wq = 8;
        ptr += p->vocab_size * p->dim * sizeof(WeightType);
        w->token_embedding_table = malloc(p->vocab_size * sizeof(float));
        #else
        w->token_embedding_table = (float*) ptr;
        w->wcls = w->token_embedding_table; // shared embedding table
        ptr += p->vocab_size * p->dim * sizeof(float);
        #endif
    }
    // Initialize Quantization Weight Scales
    #ifdef QUANTIZED
    ptr += init_quant_weight_scale(w->wq, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wk, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wv, ptr, n_layers);
    ptr += init_quant_weight_scale(w->wo, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w1, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w2, ptr, n_layers);
    ptr += init_quant_weight_scale(w->w3, ptr, n_layers);
    if(shared_weights){
        w->wcls.scale = *(float*)ptr;
        ptr += sizeof(float);
    }
    #endif

    return;
}

void read_checkpoint(Config* config, TransformerWeights* weights, TransformerQScheme* qsheme) {
    char* ptr = (char*)llama_model_start;
    uint32_t magic_number = *(uint32_t*)ptr;
    ptr += sizeof(uint32_t);
    if (magic_number != 0x616b3432) { printf("Bad magic number\n"); exit(EXIT_FAILURE); }
    int version = *(int*)(ptr);
    if (version != 1) { printf( "Bad version %d, need version1\n", version); exit(EXIT_FAILURE); }
    ptr += sizeof(int);
    int header_size = 256;
    if (memcpy(config, ptr, sizeof(Config)) == NULL) { exit(EXIT_FAILURE); }
    ptr += sizeof(Config);
    printf("Model Config:\ndim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
           config->dim, config->hidden_dim, config->n_layers, config->n_heads, config->n_kv_heads, config->vocab_size, config->seq_len);

    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    printf("Initializing Weights...\n");
    void* weights_ptr = (char*)llama_model_start + header_size;
    weights_ptr = init_float_params(weights, config, weights_ptr, shared_weights);
    #ifdef QUANTIZED
    printf("Initializing Quantized Weights...\n");
    weights_ptr = init_qschemes(qsheme, config, weights_ptr);
    unsigned char pad = *(unsigned char*)weights_ptr;
    weights_ptr += pad;
    #endif
    printf("Mapping Weights...\n");
    memory_map_weights(weights, qsheme, config, weights_ptr, shared_weights);
    printf("Checkpoint Loaded...\n");
}

static int hex2int(char c){
    if(c >= '0' && c <= '9'){
        return c - '0';
    }else if(c >= 'a' && c <= 'f'){
        return c - 'a' + 10;
    }else if(c >= 'A' && c <= 'F'){
        return c - 'A' + 10;
    }
    return 0;
}

int tokscanf(const char* piece, unsigned char* byte_val){
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>'){
        return 0;
    }
    char h = piece[3];
    char l = piece[4];
    *byte_val = (hex2int(h) << 4) | hex2int(l);
    return 1;
}

void build_transformer(Transformer *t) {
    printf("Building Transformer model...\n");
    read_checkpoint(&t->config, &t->weights, &t->qscheme);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state);
}

long RMSNORM_TIMER = 0;
long FMATMUL_TIMER = 0;
long ATTENTION_TIMER = 0;
long ROPE_TIMER = 0;
long ONCHIP_KV_TIMER = 0;
long ATTN_QINT8_TIMER = 0;
long ATTN_QK_DOT_TIMER = 0;
long ATTN_AV_ACC_TIMER = 0;

void init_timers() {
    SOFTMAX_TIMER = 0;
    QMATMUL_TIMER = 0;
    QUANT_TIMER = 0;
    RMSNORM_TIMER = 0;
    FMATMUL_TIMER = 0;
    SOFTMAX_TIMER = 0;
    ATTENTION_TIMER = 0;
    ROPE_TIMER = 0;
    ONCHIP_KV_TIMER = 0;
    ATTN_QINT8_TIMER = 0;
    ATTN_QK_DOT_TIMER = 0;
    ATTN_AV_ACC_TIMER = 0;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    long start = MiCo_time();
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    long end = MiCo_time();
    RMSNORM_TIMER += end - start;
}

void fmatmul(float* xout, float* x, Tensor2D_F32* w, int n, int d) {
    long start = MiCo_time();
    Tensor2D_F32 Tx = { .shape = {1, n}, .data = x };
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };
    Tensor2D_F32 Ty = { .shape = {1, d}, .data = xout };

    MiCo_linear_f32(&Ty, &Tx, w, &Tb);
    long end = MiCo_time();
    FMATMUL_TIMER += end - start;
}

void qmatmul(float* xout, float* x, Tensor2D_Q8* w, int n, int d, 
        qtype wq, qtype aq) {
    Tensor2D_F32 Tx = { .shape = {1, n}, .data = x };
    Tensor1D_F32 Tb = { .shape = {0}, .data = NULL };
    Tensor2D_F32 Ty = { .shape = {1, d}, .data = xout };
    MiCo_bitlinear_f32(&Ty, &Tx, w, &Tb, wq, aq, 1);
}

float* forward(Transformer* transformer, int token, int pos) {

    init_timers();

    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    TransformerQScheme* qscheme = &transformer->qscheme;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int head_pairs = head_size / 2;


    MiCo_MHA_Config mha_config = {
        .n_heads = p->n_heads,
        .head_size = head_size,
        .seq_len = p->seq_len,
        .kv_dim = kv_dim,
        .kv_mul = kv_mul
    };

    long forward_start = MiCo_time();
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        #ifdef RISCV_VEXII
        printf("Processing Layer %d/%d\n", (int)l+1, (int)p->n_layers);
        #endif
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        int loff = l * p->seq_len * kv_dim;
        #ifdef USE_INT8_KV
        kv_type* qk_ptr = s->key_cache + loff + pos * kv_dim;
        kv_type* qv_ptr = s->value_cache + loff + pos * kv_dim;
        #else
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        #endif
        #ifdef QUANTIZED
        qmatmul(s->q, s->xb, w->wq + l, dim, dim,
            qscheme->wq_qtype[2*l], qscheme->wq_qtype[2*l+1]);
        qmatmul(s->k, s->xb, w->wk + l, dim, kv_dim,
            qscheme->wk_qtype[2*l], qscheme->wk_qtype[2*l+1]);
        qmatmul(s->v, s->xb, w->wv + l, dim, kv_dim,
            qscheme->wv_qtype[2*l], qscheme->wv_qtype[2*l+1]);
        #else
        fmatmul(s->q, s->xb, w->wq + l, dim, dim);
        fmatmul(s->k, s->xb, w->wk + l, dim, kv_dim);
        fmatmul(s->v, s->xb, w->wv + l, dim, kv_dim);
        #endif
        
        #ifdef USE_INT8_KV
        long quant_start = MiCo_time();
        s->key_scales[l * p->seq_len + pos] = __FP32toQ8_hw(qk_ptr, s->k, kv_dim);
        s->value_scales[l * p->seq_len + pos] = __FP32toQ8_hw(qv_ptr, s->v, kv_dim);
        QUANT_TIMER += MiCo_time() - quant_start;
        #endif

        long rope_start = MiCo_time();
        for (int i = 0; i < dim; i+=2) {

            int kpair = (i % head_size) >> 1;
            size_t ridx = (size_t)pos * head_pairs + kpair;
            float fcr = s->rope_cos[0];
            float fci = s->rope_sin[0];

            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        ROPE_TIMER += MiCo_time() - rope_start;
        long attn_start = MiCo_time();

        Tensor2D_F32 output = {
            .shape = {p->n_heads, head_size},
            .data = s->xb
        };
        Tensor2D_F32 query = {
            .shape = {p->n_heads, head_size},
            .data = s->q };

        #ifdef USE_INT8_KV
        MiCo_multihead_attention_f32_kv8(
            &output,
            &query,
            s->key_cache + loff,
            s->value_cache + loff,
            s->key_scales + l * p->seq_len,
            s->value_scales + l * p->seq_len,
            s->att,
            pos,
            &mha_config
        );
        #else
        MiCo_multihead_attention_f32(
            &output,
            &query,
            s->key_cache + loff,
            s->value_cache + loff,
            s->att,
            pos,
            &mha_config
        );
        #endif

        ATTENTION_TIMER += MiCo_time() - attn_start;

        #ifdef QUANTIZED
        qmatmul(s->xb2, s->xb, w->wo + l, dim, dim,
            qscheme->wo_qtype[2*l], qscheme->wo_qtype[2*l+1]);
        #else
        fmatmul(s->xb2, s->xb, w->wo + l, dim, dim);
        #endif
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        #ifdef QUANTIZED
        qmatmul(s->hb, s->xb, w->w1 + l, dim, hidden_dim,
            qscheme->w1_qtype[2*l], qscheme->w1_qtype[2*l+1]);
        qmatmul(s->hb2, s->xb, w->w3 + l, dim, hidden_dim,
            qscheme->w3_qtype[2*l], qscheme->w3_qtype[2*l+1]);
        #else
        fmatmul(s->hb, s->xb, w->w1 + l, dim, hidden_dim);
        fmatmul(s->hb2, s->xb, w->w3 + l, dim, hidden_dim);
        #endif
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        #ifdef QUANTIZED
        qmatmul(s->xb, s->hb, w->w2 + l, hidden_dim, dim,
            qscheme->w2_qtype[2*l], qscheme->w2_qtype[2*l+1]);
        #else
        fmatmul(s->xb, s->hb, w->w2 + l, hidden_dim, dim);
        #endif
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    long layer_time = MiCo_time() - forward_start;
    long layer_matmul_time = QMATMUL_TIMER;
    long layer_quant_time = QUANT_TIMER;
    rmsnorm(x, x, w->rms_final_weight, dim);
    memset(s->logits, 0, p->vocab_size * sizeof(float));
    long forward_end = MiCo_time();
    #ifdef RISCV_VEXII
    long prefill_time = forward_end - forward_start;
    printf("Prefill Time: %ld \n", prefill_time);
    printf("QMatMul Time: %ld \n", QMATMUL_TIMER);
    printf("Quant Time: %ld \n", QUANT_TIMER);
    printf("Attention Time: %ld \n", ATTENTION_TIMER);
    printf("RMSNorm Time: %ld \n", RMSNORM_TIMER);
    printf("RoPE Time: %ld \n", ROPE_TIMER);
    printf("Softmax Time: %ld \n", SOFTMAX_TIMER);
    printf("OnChip KV Load Time: %ld \n", ONCHIP_KV_TIMER);
    printf("Attention QInt8 Time: %ld \n", ATTN_QINT8_TIMER);
    printf("Attention QK Dot Time: %ld \n", ATTN_QK_DOT_TIMER);
    printf("Attention AV Acc Time: %ld \n", ATTN_AV_ACC_TIMER);

    const int nlayers = LLAMA_LAYERS;

    long estimated_end2end = layer_time * nlayers + (prefill_time - layer_time);
    printf("Estimated End2End Time %ld \n", estimated_end2end);
    long end2end_matmul_time = layer_matmul_time * nlayers + (QMATMUL_TIMER - layer_matmul_time);
    printf("Estimated End2End MatMul Time %ld \n", end2end_matmul_time);
    long end2end_attention_time = ATTENTION_TIMER * nlayers;
    printf("Estimated End2End Attention Time %ld \n", end2end_attention_time);
    long end2end_quant_time = layer_quant_time * nlayers + (QUANT_TIMER - layer_quant_time);
    printf("Estimated End2End Quant Time %ld \n", end2end_quant_time);
    #endif
    return s->logits;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int start_pos, int steps);
int main(){
    printf("MiCo Transformer Demo\n");
#ifdef VERIFY_QUANT_HW
    {
        float test_x[8] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 127.0f, -128.0f, 0.001f};
        int8_t qx1[8], qx2[8];
        float s1 = __FP32toQ8((qbyte*)qx1, test_x, 8);
        float s2 = __FP32toQ8_hw((qbyte*)qx2, test_x, 8);
        int mismatch = 0;
        for(int i = 0; i < 8; i++){
            if(qx1[i] != qx2[i]){
                printf("QUANT MISMATCH i=%d ref=%d hw=%d\n", i, qx1[i], qx2[i]);
                mismatch = 1;
            }
        }
        float sdiff = s1 - s2; if(sdiff<0) sdiff=-sdiff;
        if(sdiff > 1e-6f)
            printf("QUANT SCALE MISMATCH ref=%f hw=%f\n", s1, s2);
        if(!mismatch)
            printf("QUANT HW VERIFY PASS\n");
    }
#endif

    float temperature = 0.0f;
    float topp = 1.0f;
    int start_pos = CONTEXT_LEN;
    int steps = start_pos + total_step;
    char *prompt = "";
    unsigned long long rng_seed = 42;

    if (rng_seed <= 0) rng_seed = (unsigned int)time_in_ms();
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    printf("Building Tokenizer and Sampler...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    printf("Generating: \n");
    generate(&transformer, &tokenizer, &sampler, prompt, start_pos, steps);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
