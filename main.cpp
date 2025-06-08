#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>
#include <map>
#include <chrono>

using namespace std;
using Clock = chrono::high_resolution_clock;

double leave_one_out_cross_validation(const vector<vector<double>>& data, const vector<int>& feature_set) {
    int n = static_cast<int>(data.size());
    int correct_count = 0;

    for (int i = 0; i < n; ++i) {
        double true_label = data[i][0];
        double best_squared_dist = numeric_limits<double>::infinity();
        double nearest_label = std::numeric_limits<double>::quiet_NaN();

        for (int k = 0; k < n; ++k) {
            if (k == i) 
                continue;

            double sq_dist = 0.0;
            for (int feat : feature_set) {
                double di = data[i][feat];
                double dk = data[k][feat];
                double diff = di - dk;
                sq_dist += diff * diff;
            }

            if (sq_dist < best_squared_dist) {
                best_squared_dist = sq_dist;
                nearest_label = data[k][0];
            }
        }
        if (nearest_label == true_label)
            ++correct_count;
    }
    double accuracy_fraction = static_cast<double>(correct_count) / n;
    return accuracy_fraction * 100.0;
}

bool load_data(const string& filename, vector<vector<double>>& data_out) {
    ifstream infile("Datasets/"+filename);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open file '" << filename << "'\n";
        return false;
    }
    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            data_out.push_back(row);
        }
    }
    infile.close();
    return true;
}

string format_feature_set(const vector<int>& feats) {
    ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < feats.size(); ++i) {
        oss << feats[i];
        if (i + 1 < feats.size()) oss << ",";
    }
    oss << "}";
    return oss.str();
}

void run_forward_selection(const vector<vector<double>>& data) {
    int num_features = static_cast<int>(data[0].size()) - 1;

    vector<int> full_set;
    full_set.reserve(num_features);
    for (int f = 1; f <= num_features; ++f)
        full_set.push_back(f);

    double full_accuracy = leave_one_out_cross_validation(data, full_set);
    cout<<fixed<<setprecision(1);
    cout<<"\nRunning nearest neighbor with all "<<num_features<<" features, using \"leave-one-out\" evaluation, I get an accuracy of "<<full_accuracy<<"%\n";
    cout << "\nBeginning search.\n";

    vector<int> current_set;
    vector<int> best_overall_set = full_set;
    double best_overall_accuracy = full_accuracy;

    for (int level = 1; level <= num_features; ++level) {
        double best_this_level_accuracy = 0.0;
        vector<int> best_this_level_set;

        for (int feat = 1; feat <= num_features; ++feat) {
            if (find(current_set.begin(), current_set.end(), feat) == current_set.end()) {
                vector<int> candidate_set = current_set;
                candidate_set.push_back(feat);
                sort(candidate_set.begin(), candidate_set.end());

                double acc = leave_one_out_cross_validation(data, candidate_set);
                cout<<"Using feature(s) "<<format_feature_set(candidate_set)<<" accuracy is "<<acc<<"%\n";

                if (acc > best_this_level_accuracy) {
                    best_this_level_accuracy = acc;
                    best_this_level_set = candidate_set;
                }
            }
        }

        cout<<"Feature set "<<format_feature_set(best_this_level_set)<<" gave an accuracy of: "<<best_this_level_accuracy<<"%\n";

        if (best_this_level_accuracy < best_overall_accuracy) {
            cout<<"Warning, accuracy has decreased! Continuing search in case of local maxima\n";
        } else {
            best_overall_accuracy = best_this_level_accuracy;
            best_overall_set = best_this_level_set;
        }
        current_set = best_this_level_set;
    }
    cout<<"\nFinished search!! The best feature subset is "<<format_feature_set(best_overall_set)<<", which has an accuracy of "<<best_overall_accuracy<<"%\n";
}

void run_backward_elimination(const vector<vector<double>>& data) {
    int num_features = static_cast<int>(data[0].size()) - 1;

    vector<int> current_set;
    current_set.reserve(num_features);
    for (int f = 1; f <= num_features; ++f) {
        current_set.push_back(f);
    }

    double full_accuracy = leave_one_out_cross_validation(data, current_set);
    cout<<fixed<<setprecision(1);
    cout<<"\nRunning nearest neighbor with all "<<num_features<<" features, I get an accuracy of "<<full_accuracy<<"%\n";
    cout<<"\nBeginning search.\n";

    vector<int> best_overall_set = current_set;
    double best_overall_accuracy = full_accuracy;

    for (int level = 1; level <= num_features; ++level) {
        double best_this_level_accuracy = 0.0;
        vector<int> best_this_level_set;

        for (size_t idx = 0; idx < current_set.size(); ++idx) {
            int feat = current_set[idx];
            vector<int> candidate_set;
            candidate_set.reserve(current_set.size() - 1);
            for (int f : current_set) {
                if (f != feat) {
                    candidate_set.push_back(f);
                }
            }
            sort(candidate_set.begin(), candidate_set.end());
            double acc = leave_one_out_cross_validation(data, candidate_set);
            cout<<"Using feature(s) "<<format_feature_set(candidate_set)<<" accuracy is "<<acc<<"%\n";
            if (acc > best_this_level_accuracy) {
                best_this_level_accuracy = acc;
                best_this_level_set = candidate_set;
            }
        }
        cout<<"Feature set "<<format_feature_set(best_this_level_set)<<" gave an accuracy of "<<best_this_level_accuracy<<"%\n";
        if (best_this_level_accuracy < best_overall_accuracy) {
            cout<<"(Warning, accuracy has decreased! Continuing search in case of local maxima)\n";
        } else {
            best_overall_accuracy = best_this_level_accuracy;
            best_overall_set = best_this_level_set;
        }
        current_set = best_this_level_set;
    }
    cout<<"\nFinished search!! The best feature subset is "<<format_feature_set(best_overall_set)<<", which has an accuracy of "<<best_overall_accuracy<<"%\n";
}

double default_rate(const vector<vector<double>>& data) {
    map<double,int> freq;
    for (auto& row : data) 
        freq[row[0]]++;
    int max_count = 0;
    for (auto& p : freq) 
        max_count = max(max_count, p.second);
    return (static_cast<double>(max_count) / data.size()) * 100.0;
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    cout<<"Welcome to Bertie Woosters Feature Selection Algorithm\n";
    cout<<"Type in the name of the file to test: ";
    string filename;
    getline(cin, filename);

    vector<vector<double>> data;
    if (!load_data(filename, data)) {
        return 1;
    }
    if (data.empty()) {
        cerr<<"Error: Loaded data is empty.\n";
        return 1;
    }

    int num_instances = static_cast<int>(data.size());
    int num_features = static_cast<int>(data[0].size()) - 1;

    cout<<"\nThis dataset has "<<num_features<<" features (Not including the class attribute), with "<<num_instances<<" instances\n";
    cout<<"\nType the number of the algorithm you want to run\n\n"<<"1) Forward Selection\n"<<"2) Backward Elimination\n\n";

    int choice = 0;
    cin>>choice;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    double base = default_rate(data);
    cout << "Default rate using no features is: " << fixed << setprecision(1) << base << "%" << endl;
    
    auto t_start = Clock::now();
    srand(static_cast<unsigned int>(time(nullptr)));
    
    if (choice == 1) {
        run_forward_selection(data);
    } else if (choice == 2) {
        run_backward_elimination(data);
    } else {
        cout<<"Invalid choice. Exiting.\n";
    }
    auto t_end = Clock::now();
    auto duration_ms = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
    cout<<"\nTotal execution time: "<<duration_ms<<" ms\n";
    return 0;
}