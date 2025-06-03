#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

int main() {
    string input_filename="breast-cancer-wisconsin.data", output_filename="preprocessed-breast-cancer-wisconsin.data";

    ifstream infile(input_filename);
    if (!infile.is_open()) {
        cerr << "Error: Could not open input file '" << input_filename << "'\n";
        return 1;
    }

    ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file '" << output_filename << "'\n";
        return 1;
    }

    string line;
    while (getline(infile, line)) {
        if (line.empty()) 
            continue;
        vector<string> tokens;
        tokens.reserve(11);
        istringstream linestream(line);
        string token;
        while (getline(linestream, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 11) {
            continue;
        }

        outfile << tokens[10];
        for (int i = 1; i <= 9; ++i) {
            outfile << " " << tokens[i];
        }
        outfile << "\n";
    }

    infile.close();
    outfile.close();

    cout << "Preprocessing complete. Written to '" << output_filename << "'.\n";
    return 0;
}