#include <fstream>
#include <vector>
#include <unordered_map>

// 32x32 Threads in a block.
#define NTHREADS_X 32
#define NTHREADS_Y 32
#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

using std::string;
using std::vector;
using std::unordered_map;

const int MAXN = static_cast<int>(1e3);

struct rule {
    string from_, to1_, to2_;

    rule(string from, string to1, string to2) : from_(from), to1_(to1), to2_(to2) {}
};

char g[MAXN][MAXN];
vector<rule> prod;

int ver_num;
bool tmp_mat[30][MAXN][MAXN];
bool* mat[30];

unordered_map<string, int> reverse_prod;
int cnt = 0;
unordered_map<string, int> bij;


/* 
* A macro used for error checking in CUDA function calls
* http://stackoverflow.com/a/14038590
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrix_mul(bool *a, bool *b, bool *c, int a_ncolumns, int c_nlines,
        int c_ncolumns, int nBlocks) {
    int i, z;
    bool cur_value = 0;

    // number of multiplications
    int nMultiplications = a_ncolumns;

    // multiplications per block
    int multiplicationsInBlock = NTHREADS_Y;

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ bool s_a[NTHREADS_Y][NTHREADS_X];
    __shared__ bool s_b[NTHREADS_Y][NTHREADS_X];

    int a_tLine, a_tColumn, b_tLine, b_tColumn;

    for (z = 0; z < nBlocks; z++) {

        // Load Matrix A
        a_tLine = (blockIdx.y * NTHREADS_Y + threadIdx.y);
        a_tColumn = (z * NTHREADS_X + threadIdx.x);
        if (a_tLine < c_nlines && a_tColumn < a_ncolumns) {
            s_a[threadIdx.y][threadIdx.x] = a[ (a_ncolumns * a_tLine) + a_tColumn];
        }

        // Load Matrix B
        b_tLine = (z * NTHREADS_Y + threadIdx.y);
        b_tColumn = (blockIdx.x * NTHREADS_X + threadIdx.x);
        if (b_tLine < a_ncolumns && b_tColumn < c_ncolumns) {
            s_b[threadIdx.y][threadIdx.x] = b[ (c_ncolumns * b_tLine) + b_tColumn ];
        }

        __syncthreads();

	    // Checkin position in Matrix C
        if (column < c_ncolumns && line < c_nlines) {
            if (nMultiplications < NTHREADS_Y) {
                multiplicationsInBlock = nMultiplications;
            }

            for (i = 0; i < multiplicationsInBlock; i++) {
                cur_value |= s_a[threadIdx.y][i] && s_b[i][threadIdx.x];
            }

            nMultiplications -= NTHREADS_Y;
        }

        __syncthreads();
    }

    // Set value to Matrix C
    if (column < c_ncolumns && line < c_nlines) {
        c[line * c_ncolumns + column] = cur_value;
    }
}

bool mult(int res, int a1, int a2) {  // res += a1 * a2
    bool *d_a, *d_b, *d_c;

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    int size = ver_num * ver_num * sizeof(bool);
    gpuErrchk( cudaMalloc((void **) &d_a, size) );
    gpuErrchk( cudaMalloc((void **) &d_b, size) );
    gpuErrchk( cudaMalloc((void **) &d_c, size) );

    bool* c = (bool*)malloc(size);
    memset(c, false, size);

    gpuErrchk( cudaMemcpy(d_a, mat[a1], size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, mat[a2], size, cudaMemcpyHostToDevice) );

    dim3 tbloco = dim3(
                    (int) std::ceil( (double) ver_num / NTHREADS_X ),
                    (int) std::ceil( (double) ver_num / NTHREADS_Y ),
                    1);

    dim3 tthreads = dim3(NTHREADS_X, NTHREADS_Y, 1);

    cudaEventRecord(start);
    // kernel call
    matrix_mul<<<tbloco,tthreads>>>(d_a, d_b, d_c, ver_num, ver_num,
        ver_num, (int) std::ceil( (double) ver_num / NTHREADS_X));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaEventSynchronize(stop) );

    bool has_changed = false;
    for (int i = 0; i < ver_num * ver_num; ++i) {
        has_changed |= (c[i] != mat[res][i]);
        mat[res][i] |= c[i];
    }
    free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return has_changed;
}

int main(int argc, char* argv[]) {
    auto nfh_stream = std::ifstream(argv[1], std::ifstream::in);
    string from, to1, to2;
    while (nfh_stream >> from >> to1) {
        if (to1[0] >= 'A' && to1[0] <= 'Z') {
            nfh_stream >> to2;
            prod.emplace_back(from, to1, to2);
            if (!bij.count(to1)) {
                bij[to1] = cnt++;
            }
            if (!bij.count(to2)) {
                bij[to2] = cnt++;
            }
        } else {
            if (!bij.count(from)) {
                bij[from] = cnt++;
            }
            reverse_prod[to1] = bij[from];
        }
    }
    nfh_stream.close();
    auto graph_stream = std::ifstream(argv[2], std::ifstream::in);
    int a, b;
    string c;
    while (graph_stream >> a >> c >> b) {
        tmp_mat[reverse_prod[c]][a - 1][b - 1] = 1;
        ver_num = max(ver_num, max(a - 1, b - 1));
    }
    ++ver_num;
    graph_stream.close();
    for (auto & term : reverse_prod) {
        mat[term.second] = (bool*)malloc(ver_num * ver_num);
        memset(mat[term.second], false, ver_num * ver_num);
        for (int i = 0; i < ver_num; ++i) {
            for (int j = 0; j < ver_num; ++j) {
                mat[term.second][i * ver_num + j] = tmp_mat[term.second][i][j];
            }
        }
    }
    // int iter_num = 1;

    // auto time_start = chrono::steady_clock::now();
    while (true) {
        bool has_changed = false;
        for (auto & cur_prod : prod) {
            has_changed |= mult(bij[cur_prod.from_], bij[cur_prod.to1_], bij[cur_prod.to2_]);
        }
        if (!has_changed) {
            break;
        }
        // ++iter_num;
    }
    // auto time_finish = chrono::steady_clock::now();
    auto out_stream = std::ofstream(argv[3], std::ifstream::out);
    for (auto& s : bij) {
        if (s.first[0] <= 'A' || s.first[0] >= 'Z') {
            continue;
        }
        out_stream << s.first << " ";
        for (int i = 0; i < ver_num; ++i) {
            for (int j = 0; j < ver_num; ++j) {
                if (mat[s.second][i * ver_num + j]) {
                    out_stream << i + 1 << " " << j + 1 << " ";
                }
            }
        }
        out_stream << '\n';
    }
    out_stream.close();
    return 0;
}
