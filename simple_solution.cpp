#include <bits/stdc++.h>

using namespace std;

const int MAXN = (int)1e3;

struct rule {
    string from_, to1_, to2_;

    rule(string from, string to1, string to2) : from_(from), to1_(to1), to2_(to2) {}
};

char g[MAXN][MAXN];
vector<rule> prod;

bool mat[30][MAXN][MAXN];

unordered_map<string, int> reverse_prod;
int cnt = 0;
unordered_map<string, int> bij;

void print(int c, int ver_num) {
    for (int i = 0; i < ver_num; ++i) {
        for (int j = 0; j < ver_num; ++j) {
            cout << mat[c][i][j] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

bool mult(int res, int a1, int a2, int ver_num) {  // res += a1 * a2
    bool has_changed = false;
    for (int i = 0; i < ver_num; ++i) {
        for (int j = 0; j < ver_num; ++j) {
            bool cur = 0;
            for (int k = 0; k < ver_num; ++k) {
                cur |= (mat[a1][i][k] && mat[a2][k][j]);
            }
            has_changed |= (mat[res][i][j] != (mat[res][i][j] | cur));
            mat[res][i][j] |= cur;
        }
    }
    return has_changed;
}

int main(int argc, char* argv[]) {
    auto nfh_stream = ifstream(argv[1], ifstream::in);
    string from, to1, to2, delim;
    while (nfh_stream >> from >> delim >> to1) {
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
    auto graph_stream = ifstream(argv[2], ifstream::in);
    int a, b;
    string c;
    int num_vertices = 0;
    while (graph_stream >> a >> b >> c) {
        mat[reverse_prod[c]][a - 1][b - 1] = 1;
        num_vertices = max(num_vertices, max(a - 1, b - 1));
    }
    graph_stream.close();
    ++num_vertices;
    // int iter_num = 1;

    // auto time_start = chrono::steady_clock::now();
    while (true) {
        bool has_changed = false;
        for (auto & cur_prod : prod) {
            has_changed |= mult(bij[cur_prod.from_], bij[cur_prod.to1_], bij[cur_prod.to2_], num_vertices);
        }
        if (!has_changed) {
            break;
        }
        // ++iter_num;
    }
    // auto time_finish = chrono::steady_clock::now();
    auto out_stream = ofstream(argv[3], ifstream::out);
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < num_vertices; ++j) {
            bool flag = true;
            for (auto& s : bij) {
                if (mat[s.second][i][j]) {
                    if (flag) {
                        out_stream << i + 1 << " " << j + 1 << " ";
                        flag = false;
                    }
                    out_stream << s.first << " ";
                }
            }
            if (!flag) { 
                out_stream << "\n";
            }
        }
    }
    // auto diff = time_finish - time_start;
    // cout << "time: " << chrono::duration <double, milli> (diff).count() << "s\niters: " << iter_num - 1;
}