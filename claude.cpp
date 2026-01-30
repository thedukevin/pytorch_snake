#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

const int boardSize = 30;
const pair<int, int> dir[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

pair<int, int> shiftPos(pair<int, int> pos, int d) {
    return {pos.first + dir[d].first, pos.second + dir[d].second};
}

bool isValid(pair<int, int> pos) {
    return pos.first >= 0 && pos.first < boardSize && pos.second >= 0 && pos.second < boardSize;
}

int manhattanDist(pair<int, int> pos1, pair<int, int> pos2) {
    return abs(pos1.first - pos2.first) + abs(pos1.second - pos2.second);
}

class Environment {
public:
    int body[boardSize][boardSize];
    pair<int, int> head;
    pair<int, int> tail;
    pair<int, int> apple;
    int time;

    Environment() {
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                body[i][j] = -1;
            }
        }
        head = {boardSize / 2, 2};
        tail = {boardSize / 2, 1};
        body[tail.first][tail.second] = 0;
        randomizeApple();
        time = 0;
    }

    Environment(const Environment& other) {
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                body[i][j] = other.body[i][j];
            }
        }
        head = other.head;
        tail = other.tail;
        apple = other.apple;
        time = other.time;
    }

    Environment& operator=(const Environment& other) {
        if (this != &other) {
            for (int i = 0; i < boardSize; i++) {
                for (int j = 0; j < boardSize; j++) {
                    body[i][j] = other.body[i][j];
                }
            }
            head = other.head;
            tail = other.tail;
            apple = other.apple;
            time = other.time;
        }
        return *this;
    }

    vector<int> validDirs() {
        vector<int> valid;
        for (int d = 0; d < 4; d++) {
            pair<int, int> newHead = shiftPos(head, d);
            if (isValid(newHead) && body[newHead.first][newHead.second] == -1) {
                valid.push_back(d);
            }
        }
        return valid;
    }

    int detAction(int d) {
        body[head.first][head.second] = d;
        head = shiftPos(head, d);
        time++;

        if (head == apple) {
            return 1;
        } else {
            int tailDir = body[tail.first][tail.second];
            body[tail.first][tail.second] = -1;
            tail = shiftPos(tail, tailDir);
            return 0;
        }
    }

    void randomizeApple() {
        vector<pair<int, int>> openSquares;
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (body[i][j] == -1 && make_pair(i, j) != head) {
                    openSquares.push_back({i, j});
                }
            }
        }
        if (openSquares.empty()) return;
        apple = openSquares[rand() % openSquares.size()];
    }

    string toString() {
        stringstream ss;
        ss << "Time: " << time << "\n";
        ss << "Head: (" << head.first << ", " << head.second << ")\n";
        ss << "Apple: (" << apple.first << ", " << apple.second << ")\n";
        int size = 1;
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (body[i][j] != -1) size++;
            }
        }
        ss << "Size: " << size << "\n";
        
        char grid[2 * boardSize - 1][2 * boardSize - 1];
        for (int i = 0; i < 2 * boardSize - 1; i++) {
            for (int j = 0; j < 2 * boardSize - 1; j++) {
                grid[i][j] = ' ';
            }
        }
        
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (make_pair(i, j) == head) {
                    grid[2 * i][2 * j] = 'H';
                } else if (body[i][j] != -1) {
                    int d = body[i][j];
                    grid[2 * i][2 * j] = 'O';
                    grid[2 * i + dir[d].first][2 * j + dir[d].second] = (d % 2 == 0) ? '-' : '|';
                } else if (make_pair(i, j) == apple) {
                    grid[2 * i][2 * j] = 'A';
                } else {
                    grid[2 * i][2 * j] = '.';
                }
            }
        }
        
        for (int i = 0; i < 2 * boardSize - 1; i++) {
            for (int j = 0; j < 2 * boardSize - 1; j++) {
                ss << grid[i][j];
            }
            ss << "\n";
        }
        return ss.str();
    }
};

struct HeuristicResult {
    double value;
    int dist;
    bool tailVis;
    double avgWaitBonus;
};

HeuristicResult heuristic(Environment& env) {
    bool visited[boardSize][boardSize] = {false};
    int retractible[boardSize][boardSize];
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++) {
            retractible[i][j] = env.body[i][j];
        }
    }
    pair<int, int> curr_tail = env.tail;

    int dist = -1;
    if (env.apple != make_pair(-1, -1)) {
        queue<pair<int, int>> q;
        q.push(env.head);
        visited[env.head.first][env.head.second] = true;

        bool finished = false;
        for (int i = 0; i < boardSize * boardSize && !finished; i++) {
            queue<pair<int, int>> next_queue;
            while (!q.empty()) {
                pair<int, int> pos = q.front();
                q.pop();
                if (pos == env.apple) {
                    dist = i;
                    finished = true;
                    break;
                }
                for (int d = 0; d < 4; d++) {
                    pair<int, int> newPos = shiftPos(pos, d);
                    if (isValid(newPos) && retractible[newPos.first][newPos.second] == -1 && 
                        !visited[newPos.first][newPos.second]) {
                        visited[newPos.first][newPos.second] = true;
                        next_queue.push(newPos);
                    }
                }
            }
            if (finished) break;
            if (curr_tail != env.head) {
                retractible[curr_tail.first][curr_tail.second] = -1;
                for (int d = 0; d < 4; d++) {
                    pair<int, int> neigh = shiftPos(curr_tail, d);
                    if (isValid(neigh) && visited[neigh.first][neigh.second]) {
                        next_queue.push(neigh);
                    }
                }
                curr_tail = shiftPos(curr_tail, env.body[curr_tail.first][curr_tail.second]);
            }
            q = next_queue;
        }
    }

    double distBonus = (dist != -1) ? -dist : 0;

    retractible[curr_tail.first][curr_tail.second] = -1;

    bool visited2[boardSize][boardSize] = {false};
    int tin[boardSize][boardSize];
    int low[boardSize][boardSize];
    bool isAP[boardSize][boardSize] = {false};
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++) {
            tin[i][j] = -1;
            low[i][j] = -1;
        }
    }

    int tarjan_timer = 0;
    
    function<void(pair<int, int>, pair<int, int>)> dfs = [&](pair<int, int> v, pair<int, int> p) {
        visited2[v.first][v.second] = true;
        tin[v.first][v.second] = low[v.first][v.second] = tarjan_timer++;
        int children = 0;
        for (int d = 0; d < 4; d++) {
            pair<int, int> newPos = shiftPos(v, d);
            if (!isValid(newPos) || retractible[newPos.first][newPos.second] != -1 || newPos == p) {
                continue;
            }
            if (visited2[newPos.first][newPos.second]) {
                low[v.first][v.second] = min(low[v.first][v.second], tin[newPos.first][newPos.second]);
            } else {
                dfs(newPos, v);
                low[v.first][v.second] = min(low[v.first][v.second], low[newPos.first][newPos.second]);
                if (low[newPos.first][newPos.second] >= tin[v.first][v.second] && p != make_pair(-1, -1)) {
                    isAP[v.first][v.second] = true;
                }
                children++;
            }
        }
        if (p == make_pair(-1, -1) && children > 1) {
            isAP[v.first][v.second] = true;
        }
    };

    dfs(env.head, {-1, -1});

    int components[boardSize][boardSize];
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++) {
            components[i][j] = -1;
        }
    }
    int componentCount = 0;

    map<int, int> componentSizes;
    map<int, vector<pair<int, int>>> compConnectedAPs;
    map<pair<int, int>, pair<int, int>> parent_pos;
    map<int, pair<int, int>> parent_comp;

    auto fillComponent = [&](pair<int, int> start, int compID) -> vector<pair<int, int>> {
        queue<pair<int, int>> q;
        q.push(start);
        int size = 1;
        vector<pair<int, int>> connectedAPs;
        while (!q.empty()) {
            pair<int, int> top = q.front();
            q.pop();
            for (int d = 0; d < 4; d++) {
                pair<int, int> newPos = shiftPos(top, d);
                if (isValid(newPos) && retractible[newPos.first][newPos.second] == -1 && 
                    components[newPos.first][newPos.second] == -1) {
                    if (!isAP[newPos.first][newPos.second]) {
                        components[newPos.first][newPos.second] = compID;
                        q.push(newPos);
                        size++;
                    } else {
                        connectedAPs.push_back(newPos);
                    }
                }
            }
        }
        componentSizes[compID] = size;
        return connectedAPs;
    };

    pair<int, int> start = (env.apple == make_pair(-1, -1)) ? env.head : env.apple;

    queue<pair<int, int>> q;
    q.push(start);
    if (isAP[start.first][start.second]) {
        components[start.first][start.second] = -2;
    } else {
        components[start.first][start.second] = componentCount;
        compConnectedAPs[componentCount] = fillComponent(start, componentCount);
        componentCount++;
    }

    while (!q.empty()) {
        pair<int, int> top = q.front();
        q.pop();
        if (isAP[top.first][top.second]) {
            for (int d = 0; d < 4; d++) {
                pair<int, int> newPos = shiftPos(top, d);
                if (isValid(newPos) && retractible[newPos.first][newPos.second] == -1 && 
                    components[newPos.first][newPos.second] == -1) {
                    q.push(newPos);
                    if (isAP[newPos.first][newPos.second]) {
                        components[newPos.first][newPos.second] = -2;
                        parent_pos[newPos] = top;
                    } else {
                        components[newPos.first][newPos.second] = componentCount;
                        compConnectedAPs[componentCount] = fillComponent(newPos, componentCount);
                        parent_comp[componentCount] = top;
                        componentCount++;
                    }
                }
            }
        } else {
            for (auto conn : compConnectedAPs[components[top.first][top.second]]) {
                if (components[conn.first][conn.second] == -1) {
                    components[conn.first][conn.second] = -2;
                    parent_pos[conn] = {components[top.first][top.second], components[top.first][top.second]};
                    q.push(conn);
                }
            }
        }
    }

    bool visited3[boardSize][boardSize] = {false};
    bool tailVis = false;
    queue<pair<int, int>> q2;
    q2.push(env.head);
    visited3[env.head.first][env.head.second] = true;
    while (!q2.empty()) {
        pair<int, int> top = q2.front();
        q2.pop();
        for (int d = 0; d < 4; d++) {
            pair<int, int> newPos = shiftPos(top, d);
            if (isValid(newPos) && env.body[newPos.first][newPos.second] == -1 && 
                !visited3[newPos.first][newPos.second]) {
                visited3[newPos.first][newPos.second] = true;
                q2.push(newPos);
            }
            if (top != env.head && newPos == env.tail) {
                tailVis = true;
            }
        }
    }

    double avgWaitBonus = 0;

    return {distBonus + avgWaitBonus, dist, tailVis, avgWaitBonus};
}

void printToFile(const char* file, const string& text) {
    if (file == nullptr) {
        cout << text << endl;
    } else {
        ofstream f(file, ios::app);
        f << text << endl;
        f.close();
    }
}

pair<int, int> sim() {
    Environment env;

    const char* outputFile = nullptr;
    // const char* outputFile = "snake.txt";  // Uncomment to enable file output

    if (outputFile != nullptr) {
        ofstream f(outputFile);
        f.close();
    }

    vector<int> safeActions;
    bool firstIteration = true;

    for (int t = 0; t < 50000; t++) {
        vector<vector<double>> vals(4);
        vector<vector<HeuristicResult>> features(4);
        vector<pair<double, pair<int, int>>> action_pairs;

        for (int d1 = 0; d1 < 4; d1++) {
            vector<int> valid = env.validDirs();
            if (find(valid.begin(), valid.end(), d1) == valid.end()) {
                vals[d1].push_back(-1e9);  // Marker for invalid
                features[d1].push_back({-1e9, -1, false, 0});
                continue;
            }
            Environment env1 = env;
            int isApple = env1.detAction(d1);
            if (isApple) {
                HeuristicResult result = heuristic(env1);
                vals[d1].push_back(result.value);
                features[d1].push_back(result);
                action_pairs.push_back({result.value, {d1, -1}});
                continue;
            }

            for (int d2 = 0; d2 < 4; d2++) {
                vector<int> valid1 = env1.validDirs();
                if (find(valid1.begin(), valid1.end(), d2) == valid1.end()) {
                    vals[d1].push_back(-1e9);  // Marker for invalid
                    features[d1].push_back({-1e9, -1, false, 0});
                    continue;
                }

                Environment env2 = env1;
                env2.detAction(d2);
                HeuristicResult result = heuristic(env2);
                vals[d1].push_back(result.value);
                features[d1].push_back(result);
                action_pairs.push_back({result.value, {d1, d2}});
            }
        }

        // Print environment, vals, and features
        printToFile(outputFile, env.toString());
        
        stringstream vals_ss;
        vals_ss << "[";
        for (int i = 0; i < 4; i++) {
            if (vals[i].size() == 1) {
                if (vals[i][0] == -1e9) {
                    vals_ss << "None";
                } else {
                    vals_ss << vals[i][0];
                }
            } else {
                vals_ss << "[";
                for (size_t j = 0; j < vals[i].size(); j++) {
                    if (vals[i][j] == -1e9) {
                        vals_ss << "None";
                    } else {
                        vals_ss << vals[i][j];
                    }
                    if (j < vals[i].size() - 1) vals_ss << ", ";
                }
                vals_ss << "]";
            }
            if (i < 3) vals_ss << ", ";
        }
        vals_ss << "]";
        printToFile(outputFile, vals_ss.str());
        
        stringstream feat_ss;
        feat_ss << "[";
        for (int i = 0; i < 4; i++) {
            if (features[i].size() == 1) {
                if (features[i][0].value == -1e9) {
                    feat_ss << "None";
                } else {
                    feat_ss << "(" << features[i][0].dist << ", " << features[i][0].tailVis << ", " << features[i][0].avgWaitBonus << ")";
                }
            } else {
                feat_ss << "[";
                for (size_t j = 0; j < features[i].size(); j++) {
                    if (features[i][j].value == -1e9) {
                        feat_ss << "None";
                    } else {
                        feat_ss << "(" << features[i][j].dist << ", " << features[i][j].tailVis << ", " << features[i][j].avgWaitBonus << ")";
                    }
                    if (j < features[i].size() - 1) feat_ss << ", ";
                }
                feat_ss << "]";
            }
            if (i < 3) feat_ss << ", ";
        }
        feat_ss << "]";
        printToFile(outputFile, feat_ss.str());

        sort(action_pairs.begin(), action_pairs.end(), greater<pair<double, pair<int, int>>>());

        int safeAct = -1;

        for (auto& p : action_pairs) {
            double val = p.first;
            int act1 = p.second.first;
            int act2 = p.second.second;

            Environment traj = env;
            vector<int> actionHist = {act1};
            traj.detAction(act1);
            if (act2 != -1) {
                actionHist.push_back(act2);
                traj.detAction(act2);
            }
            bool safe = false;
            for (int i = 0; i < 300; i++) {
                if (traj.validDirs().empty()) {
                    break;
                }
                HeuristicResult result = heuristic(traj);
                if (result.tailVis) {
                    safe = true;
                    break;
                }

                double bestVal = -1e9;
                int bestAct = -1;
                for (int d : traj.validDirs()) {
                    Environment curr_env = traj;
                    curr_env.detAction(d);
                    HeuristicResult hres = heuristic(curr_env);
                    if (hres.value > bestVal) {
                        bestVal = hres.value;
                        bestAct = d;
                    }
                }
                traj.detAction(bestAct);
                actionHist.push_back(bestAct);
                if (traj.head == traj.apple) {
                    traj.apple = {-1, -1};
                }
            }
            if (safe) {
                safeAct = act1;
                safeActions = actionHist;
                break;
            }
        }

        if (safeAct == -1) {
            if (firstIteration || safeActions.empty()) {
                // No safe path found and no backup actions - game over
                break;
            }
            for (int act : safeActions) {
                env.detAction(act);
            }
        } else {
            firstIteration = false;
            if (safeActions.size() == 2) {
                env.detAction(safeAct);
                safeActions.erase(safeActions.begin());
            } else {
                for (size_t i = 0; i < safeActions.size(); i++) {
                    env.detAction(safeActions[0]);
                    safeActions.erase(safeActions.begin());
                    if (env.head == env.apple) {
                        env.randomizeApple();
                        break;
                    }
                }
            }
        }

        if (env.validDirs().empty()) {
            break;
        }
    }

    int size = 1;
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++) {
            if (env.body[i][j] != -1) size++;
        }
    }
    return {size, env.time};
}

int main() {
    srand(42);

    vector<int> sizes;
    vector<int> lengths;
    vector<bool> wins;

    for (int i = 0; i < 1; i++) {
        auto result = sim();
        int size = result.first;
        int length = result.second;
        cout << size << " " << length << endl;
        sizes.push_back(size);
        lengths.push_back(length);
        bool win = (size == boardSize * boardSize);
        wins.push_back(win);
    }

    return 0;
}