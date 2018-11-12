#include <bits/stdc++.h>

using namespace std;

const int MAXN = (int)1e3;

struct rule {
    char from_, to1_, to2_;

    rule(char from, char to1, char to2) : from_(from), to1_(to1), to2_(to2) {}
};

char g[MAXN][MAXN];
vector<rule> prod;

bool mat[30][MAXN][MAXN];

unordered_map<char, char> reverse_prod;
unordered_set<char> nonterm;

void print(char c, int ver_num) {
    for (int i = 0; i < ver_num; ++i) {
        for (int j = 0; j < ver_num; ++j) {
            cout << mat[c - 'A'][i][j] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

bool mult(char res, char a1, char a2, int ver_num) {  // res += a1 * a2
    bool has_changed = false;
    for (int i = 0; i < ver_num; ++i) {
        for (int j = 0; j < ver_num; ++j) {
            bool cur = 0;
            for (int k = 0; k < ver_num; ++k) {
                cur |= (mat[a1 - 'A'][i][k] && mat[a2 - 'A'][k][j]);
            }
            has_changed |= (mat[res - 'A'][i][j] != (mat[res - 'A'][i][j] | cur));
            mat[res - 'A'][i][j] |= cur;
        }
    }
    return has_changed;
}

int main() {
    auto nfh_stream = ifstream("nfh.txt", ifstream::in);
    string to;
    char from;
    while (nfh_stream >> from >> to) {
        if (to[0] >= 'A' && to[0] <= 'Z') {
            prod.emplace_back(from, to[0], to[1]);
            nonterm.insert(to[0]);
            nonterm.insert(to[1]);
        } else {
            reverse_prod[to[0]] = from;
        }
    }
    nfh_stream.close();
    auto graph_stream = ifstream("graph.txt", ifstream::in);
    int a, b;
    char c;
    int num_vertices = 0;
    while (graph_stream >> a >> b >> c) {
        mat[reverse_prod[c] - 'A'][a][b] = 1;
        num_vertices = max(num_vertices, max(a, b));
    }
    graph_stream.close();
    ++num_vertices;
    int iter_num = 1;

    auto time_start = chrono::steady_clock::now();
    while (true) {
        bool has_changed = false;
        for (auto & cur_prod : prod) {
            has_changed |= mult(cur_prod.from_, cur_prod.to1_, cur_prod.to2_, num_vertices);
        }
        if (!has_changed) {
            break;
        }
        ++iter_num;
    }
    auto time_finish = chrono::steady_clock::now();
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < num_vertices; ++j) {
            cout << "{";
            for (char c : nonterm) {
                if (mat[c - 'A'][i][j]) {
                    cout << c << ',';
                }
            }
            cout << "}\t";
        }
        cout << '\n';
    }
    auto diff = time_finish - time_start;
    cout << "time: " << chrono::duration <double, milli> (diff).count() << "s\niters: " << iter_num - 1;
    cout << endl;
}