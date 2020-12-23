#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdint.h>


#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void gpu_multiplication(double* vectorIN, bool* vectorBIN, size_t dimension) {
	
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ double s[1024];
	s[0] = 1;
	__syncthreads();


	for (size_t i = index; i < dimension; i += blockDim.x * gridDim.x) {
		if (vectorBIN[i]) {
			s[index] *= vectorIN[i];
		}
		else {
			s[index] *= (1 - vectorIN[i]);
		}
		
	}
	__syncthreads();

	if (index == 0) {
		for (size_t i = 1; i < 1024; i++) {
			if (vectorBIN[i]) {
				s[index] *= vectorIN[i];
			}
			else {
				s[index] *= (1 - vectorIN[i]);
			}
		}
		vectorIN[0] = s[0];
	}

}



fstream spam_file;
fstream nonspam_file;
vector<string> words;
vector<int> spam_occ;
vector<int> nonspam_occ;

int * spam_occurencies;
int * nonspam_occurencies;

string * word_vector;
double * spam_vector;
double * nonspam_vector;
bool * binary_vector;

int spam_length;
int nonspam_length;

void print_words_and_occurencies() {
	for (size_t i = 0; i < words.size(); i++) {
		cout << words[i] << " " << spam_occ[i] << " " << nonspam_occ[i] << endl;
	}
}

void print_vectors() {
	cout << "spam_length: " << spam_length << ", nonspamlength: " << nonspam_length << endl;
	cout << "word_vector" << " " << "spam_vector" << " " << "nonspam_vector" << endl;
	for (int i = 0; i < words.size(); i++) {
		cout << word_vector[i] << " " << spam_vector[i] << " " << nonspam_vector[i] << endl;
	}
}

void finish_vectors() {

	word_vector = new string[words.size()];
	spam_vector = new double[words.size()];
	nonspam_vector = new double[words.size()];

	for (int i = 0; i < words.size(); i++) {
		word_vector[i] = words[i];
		spam_vector[i] = (double) spam_occ[i] / (double) spam_length;
		nonspam_vector[i] = (double) nonspam_occ[i] / (double) nonspam_length;
	}

}

void count_occurencies_length() {
	string line;
	int counter = -1;

	while (!spam_file.eof()) {

		counter++;
		getline(spam_file, line);

	}

	spam_length = counter;
	counter = -1;
	while (!nonspam_file.eof()) {

		counter++;
		getline(nonspam_file, line);

	}
	nonspam_length = counter;
	
	spam_file.clear();
	spam_file.seekg(0, ios::beg);
	nonspam_file.clear();
	nonspam_file.seekg(0, ios::beg);

}


void add_spam_training_examples() {
	cout << "Add spam examples to the database (\"quit\" to exit function.)" << endl;
	string spam_example = "";
	cin.ignore();
	do {
		getline(cin, spam_example);
		if (spam_example.compare("quit")) {
			spam_file << spam_example << endl;
		}
		else {
			break;
		}
	} while (spam_example.compare("quit"));
	spam_file.clear();
	spam_file.seekg(0, ios::beg);
	nonspam_file.clear();
	nonspam_file.seekg(0, ios::beg);
}

void add_nonspam_training_examples() {
	cout << "Add nonspam examples to the database (\"quit\" to exit function.)" << endl;
	string nonspam_example = "";
	cin.ignore();
	do {
		getline(cin, nonspam_example);
		if (nonspam_example.compare("quit")) {
			nonspam_file << nonspam_example << endl;
		}
		else {
			break;
		}
	} while (nonspam_example.compare("quit"));
	cout << "haha" << endl;
	spam_file.clear();
	spam_file.seekg(0, ios::beg);
	nonspam_file.clear();
	nonspam_file.seekg(0, ios::beg);

}

void create_words_vector() {

	words.clear();

	count_occurencies_length();
	string word;
	string line;
	string separator = " ";
	size_t position;
	
	while (!spam_file.eof()) {

		getline(spam_file, line);
		position = line.find(separator);
		while (line.compare("")==1) {
			position = line.find(separator);
			word = line.substr(0, position);
			line.erase(0, position+1);
			if (find(words.begin(), words.end(), word) != words.end()) {
				/* words contains word */
				/* words contains word */
				int pos = distance(words.begin(), find(words.begin(), words.end(), word));
				spam_occ[pos] += 1;
			}
			else {
				/* words does not contain word */
				words.push_back(word);
				int pos = distance(words.begin(), find(words.begin(), words.end(), word));
				spam_occ.push_back(1);
			}
			if (position == string::npos) {
				break;
			}
		}

		
	}
	
	
	for (int i = 0; i < words.size(); i++) {
		nonspam_occ.push_back(0);
	}
	
	while (!nonspam_file.eof()) {

		getline(nonspam_file, line);

		position = line.find(separator);
		while (line.compare("") == 1) {
			position = line.find(separator);
			word = line.substr(0, position);
			line.erase(0, position + 1);
			if (find(words.begin(), words.end(), word) != words.end()) {
				/* words contains word */
				int pos = distance(words.begin(), find(words.begin(), words.end(), word));
				nonspam_occ[pos] += 1;
			}
			else {
				/* words does not contain word */
				words.push_back(word);
				int pos = distance(words.begin(), find(words.begin(), words.end(), word));
				nonspam_occ.push_back(1);
				spam_occ.push_back(0);

			}

			if (position == string::npos) {
				break;
			}
		}
	}
	spam_file.clear();
	spam_file.seekg(0, ios::beg);
	nonspam_file.clear();
	nonspam_file.seekg(0, ios::beg);

	finish_vectors();
	
}

void run_cpu() {
	double p_test_spam = 1.0;
	double p_test_nonspam = 1.0;
	for (int i = 0; i < words.size(); i++) {
		if (binary_vector[i]) {
			p_test_spam *= spam_vector[i];
			p_test_nonspam *= nonspam_vector[i];
		}
		else {
			p_test_spam *= (1.0-spam_vector[i]);
			p_test_nonspam *= (1.0 - nonspam_vector[i]);
		}
	}

	double p_spam = (double) spam_length / (double) (spam_length + nonspam_length);
	double p_nonspam = (double) nonspam_length / (double) (spam_length + nonspam_length);
	double p_spam_test = 0.0;
	double p_nonspam_test = 0.0;

	p_spam_test = (p_test_spam * p_spam) / (p_test_nonspam  * p_nonspam + p_test_spam * p_spam);

	p_nonspam_test = (p_test_nonspam * p_nonspam) / (p_test_nonspam  * p_nonspam + p_test_spam * p_spam);


	cout << "P(spam) = " << p_spam << endl;
	cout << "P(nonspam) = " << p_nonspam << endl;
	cout << "P(test|spam) = " << p_test_spam << endl;
	cout << "P(test|nonspam) = " << p_test_nonspam << endl;
	cout << "P(spam|test) = " << p_spam_test << endl;
	cout << "P(nonspam|test) = " << p_nonspam_test << endl;

	if (p_spam_test > p_nonspam) {
		cout << "Message is a spam." << endl;
	}
	else {
		cout << "Message is not a spam." << endl;
	}

}

void run_gpu() {
	double p_test_spam = 1.0;
	double p_test_nonspam = 1.0;
	
	// ------------------------------- CUDA -----------------------

	int dim = words.size();

	size_t double_array_dim = dim * sizeof(double);
	size_t bool_array_dim = dim * sizeof(bool);
	//size_t array_dim = dim * sizeof(int64_t);

	double *vector = (double *)malloc(double_array_dim);
	bool *vector_binary = (bool *)malloc(bool_array_dim);
	double * dev_array;
	bool * dev_binary;

	cudaMalloc(&dev_array, double_array_dim);
	cudaMalloc(&dev_binary, bool_array_dim);

	
	
	// NONSPAM_VECTOR

	// HOST ----> DEVICE
	if (cudaMemcpy(dev_array, nonspam_vector, double_array_dim, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not be copied! 1" << endl;
		return;
	}

	if (cudaMemcpy(dev_binary, binary_vector, bool_array_dim, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not be copied! 2" << endl;
		return;
	}

	int blockSize = 1024;
	gpu_multiplication << <1, blockSize >> >(dev_array, dev_binary, dim);
	cudaDeviceSynchronize();

	// DEVICE ---> HOST
	if (cudaMemcpy(vector, dev_array, sizeof(double) * dim, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Could not be copied! 3" << endl;
		return;
	}
	p_test_nonspam	= vector[0];

	
	
	// SPAM_VECTOR

	// HOST ----> DEVICE
	if (cudaMemcpy(dev_array, spam_vector, double_array_dim, cudaMemcpyHostToDevice) != cudaSuccess) {
		cout << "Could not be copied! 4" << endl;
		return;
	}

	gpu_multiplication << <1, blockSize >> >(dev_array, dev_binary, dim);
	cudaDeviceSynchronize();

	// DEVICE ---> HOST
	if (cudaMemcpy(vector, dev_array, sizeof(double) * dim, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Could not be copied! 5" << endl;
		return;
	}
	p_test_spam = vector[0];

	free(vector);
	cudaFree(dev_array);
	cudaFree(dev_binary);
	// -------------------------------- CUDA END ------------------

	double p_spam = (double)spam_length / (double)(spam_length + nonspam_length);
	double p_nonspam = (double)nonspam_length / (double)(spam_length + nonspam_length);
	double p_spam_test = 0.0;
	double p_nonspam_test = 0.0;

	p_spam_test = (p_test_spam * p_spam) / (p_test_nonspam  * p_nonspam + p_test_spam * p_spam);

	p_nonspam_test = (p_test_nonspam * p_nonspam) / (p_test_nonspam  * p_nonspam + p_test_spam * p_spam);


	cout << "P(spam) = " << p_spam << endl;
	cout << "P(nonspam) = " << p_nonspam << endl;
	cout << "P(test|spam) = " << p_test_spam << endl;
	cout << "P(test|nonspam) = " << p_test_nonspam << endl;
	cout << "P(spam|test) = " << p_spam_test << endl;
	cout << "P(nonspam|test) = " << p_nonspam_test << endl;

	if (p_spam_test > p_nonspam) {
		cout << "Message is a spam." << endl;
	}
	else {
		cout << "Message is not a spam." << endl;
	}

}


void run_naive_bayes(bool gpu) {

	binary_vector = new bool[words.size()];
	for (int i = 0; i < words.size(); i++)
		binary_vector[i] = false;

	string example;
	string word;
	string separator = " ";
	size_t position;
	cout << "Input test sample: ";
	cin.ignore();
	getline(cin, example);
	int known_words = 0;
	int unknown_words = 0;

	while (example.compare("") == 1) {
		position = example.find(separator);
		word = example.substr(0, position);
		example.erase(0, position + 1);
		// Add word to vector of words
		if (find(words.begin(), words.end(), word) != words.end()) {
			/* words contains word */
			int pos = distance(words.begin(), find(words.begin(), words.end(), word));
			binary_vector[pos] = true;
			known_words++;
		}
		else {
			/* words does not contain word */
			unknown_words++;
		}

		if (position == string::npos) {
			break;
		}
	}
	cout << "known_words: " << known_words << ", unkwnown_words: " << unknown_words << endl;

	if (gpu) {
		run_gpu();
	}
	else {
		run_cpu();
	}
}



int main()
{
	// Init
	bool perform = true;
	bool ready = false;
	spam_file.open("spam.txt", ios::in | ios::out | ios::app);
	nonspam_file.open("nonspam.txt", ios::in | ios::out | ios::app);
	if (spam_file.good()==false) {
		cout << "Couldnt open spam_file.";
		getchar();
		return 0;
	}
	if (nonspam_file.good() == false) {
		cout << "Couldnt open nonspam_file.";
		getchar();
		return 0;
	}
	
	cout << "Naive Bayes Classifier as a anti-spam filter" << endl;


	int option;
	while (perform) {
		cout << endl << " ---- Choose action (number): ---- " << endl;
		cout << "1. Add spam training examples. " << endl;
		cout << "2. Add non spam training examples. " << endl;
		cout << "3. Init Naive Bayes Algorithm." << endl;
		cout << "4. Run Naive Bayes Classifier (CPU)." << endl;
		cout << "5. Run Naive Bayes Classifier (GPU)." << endl;
		cout << "6. Print words and their occurencies." << endl;
		cout << "7. Print words and vectors" << endl;
		cout << "8. Exit." << endl;
		cin >> option;
		switch (option) {
		case 1:
			add_spam_training_examples();
			ready = false;
			break;
		case 2:
			add_nonspam_training_examples();
			ready = false;
			break;
		case 3:
			create_words_vector();
			ready = true;
			break;
		case 4:
			if (ready) {
				run_naive_bayes(false);
			}
			else {
				cout << "Naive Bayes not initialized!" << endl;
			}
			break;
		case 5:
			if (ready) {
				run_naive_bayes(true);
			}
			else {
				cout << "Naive Bayes not initialized!" << endl;
			}
			break;
		case 6:
			print_words_and_occurencies();
			break;
		case 7:
			print_vectors();
			break;
		case 8:
			perform = false;
			break;
		default:
			cout << "Wrong!" << endl;
		}
	}

	// close txt files
	spam_file.close();
	nonspam_file.close();
	// delete tabs
	delete [] spam_occurencies;
	delete [] nonspam_occurencies;
	delete [] word_vector;
	delete [] spam_vector;
	delete [] nonspam_vector;

	return 0;

}