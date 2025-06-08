#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

int main() {
    string input_filename = "Datasets/breast-cancer-wisconsin.data";
    string output_filename = "Datasets/preprocessed-breast-cancer-wisconsin.data";

    ifstream infile(input_filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open input file '" << input_filename << "'\n";
        return 1;
    }

    vector<vector<double>> data;
    vector<int> labels;
    string line;

    while (getline(infile, line)) {
        if (line.empty())
            continue;

        vector<string> tokens;
        istringstream linestream(line);
        string token;

        while (getline(linestream, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 11 || tokens[10].empty())
            continue;

        vector<double> row;
        bool valid = true;
        for (int i = 1; i <= 9; ++i) {
            if (tokens[i] == "?") {
                valid = false;
                break;
            }
            row.push_back(stod(tokens[i]));
        }

        if (valid) {
            data.push_back(row);
            labels.push_back(stoi(tokens[10]));
        }
    }
    infile.close();

    vector<double> mean(9, 0.0), stddev(9, 0.0);
    int n = data.size();

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < n; ++j) {
            mean[i] += data[j][i];
        }
        mean[i] /= n;
    }

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < n; ++j) {
            stddev[i] += pow(data[j][i] - mean[i], 2);
        }
        stddev[i] = sqrt(stddev[i] / n);
    }

    ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file '" << output_filename << "'\n";
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        outfile << labels[i];
        for (int j = 0; j < 9; ++j) {
            double normalized = (data[i][j] - mean[j]) / stddev[j];
            outfile << " " << normalized;
        }
        outfile << "\n";
    }

    outfile.close();
    cout << "Preprocessing and normalization complete. Written to '" << output_filename << "'.\n";
    return 0;
}