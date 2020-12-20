#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_Conv = Layer(5*5, 6, 24*24*6);
static Layer l_sharedConv = Layer(4*4, 1, 6*6*6);
static Layer l_FC = Layer(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

	loaddata();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_Conv.clear();
	l_sharedConv.clear();
	l_FC.clear();

	clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);
	
	fp_preact_Conv<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_Conv.preact, (float (*)[5][5])l_Conv.weight);
	fp_bias_Conv<<<64, 64>>>((float (*)[24][24])l_Conv.preact, l_Conv.bias);
	apply_step_function<<<64, 64>>>(l_Conv.preact, l_Conv.output, l_Conv.O);

	fp_preact_sharedConv<<<64, 64>>>((float (*)[24][24])l_Conv.output, (float (*)[6][6])l_sharedConv.preact, (float (*)[4][4])l_sharedConv.weight);
	fp_bias_sharedConv<<<64, 64>>>((float (*)[6][6])l_sharedConv.preact, l_sharedConv.bias);
	apply_step_function<<<64, 64>>>(l_sharedConv.preact, l_sharedConv.output, l_sharedConv.O);

	fp_preact_FC<<<64, 64>>>((float (*)[6][6])l_sharedConv.output, l_FC.preact, (float (*)[6][6][6])l_FC.weight);
	fp_bias_FC<<<64, 64>>>(l_FC.preact, l_FC.bias);
	apply_step_function<<<64, 64>>>(l_FC.preact, l_FC.output, l_FC.O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();

	bp_weight_FC<<<64, 64>>>((float (*)[6][6][6])l_FC.d_weight, l_FC.d_preact, (float (*)[6][6])l_sharedConv.output);
	bp_bias_FC<<<64, 64>>>(l_FC.bias, l_FC.d_preact);

	bp_output_sharedConv<<<64, 64>>>((float (*)[6][6])l_sharedConv.d_output, (float (*)[6][6][6])l_FC.weight, l_FC.d_preact);
	bp_preact_sharedConv<<<64, 64>>>((float (*)[6][6])l_sharedConv.d_preact, (float (*)[6][6])l_sharedConv.d_output, (float (*)[6][6])l_sharedConv.preact);
	bp_weight_sharedConv<<<64, 64>>>((float (*)[4][4])l_sharedConv.d_weight, (float (*)[6][6])l_sharedConv.d_preact, (float (*)[24][24])l_Conv.output);
	bp_bias_sharedConv<<<64, 64>>>(l_sharedConv.bias, (float (*)[6][6])l_sharedConv.d_preact);

	bp_output_Conv<<<64, 64>>>((float (*)[24][24])l_Conv.d_output, (float (*)[4][4])l_sharedConv.weight, (float (*)[6][6])l_sharedConv.d_preact);
	bp_preact_Conv<<<64, 64>>>((float (*)[24][24])l_Conv.d_preact, (float (*)[24][24])l_Conv.d_output, (float (*)[24][24])l_Conv.preact);
	bp_weight_Conv<<<64, 64>>>((float (*)[5][5])l_Conv.d_weight, (float (*)[24][24])l_Conv.d_preact, (float (*)[28])l_input.output);
	bp_bias_Conv<<<64, 64>>>(l_Conv.bias, (float (*)[24][24])l_Conv.d_preact);


	apply_grad<<<64, 64>>>(l_FC.weight, l_FC.d_weight, l_FC.M * l_FC.N);
	apply_grad<<<64, 64>>>(l_sharedConv.weight, l_sharedConv.d_weight, l_sharedConv.M * l_sharedConv.N);
	apply_grad<<<64, 64>>>(l_Conv.weight, l_Conv.d_weight, l_Conv.M * l_Conv.N);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}

static void learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float train_error;
	int numEpochs = 50;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Training Started!\n");

	while (numEpochs-- > 0) {
		train_error = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_FC.bp_clear();
			l_sharedConv.bp_clear();
			l_Conv.bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_FC.d_preact, l_FC.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_FC.d_preact, 1, &tmp_err);
			train_error += tmp_err;

			time_taken += back_pass();
		}

		train_error /= train_cnt;
		fprintf(stdout, "Epoch %d => Training Error: %f, Execution GPU Time: %lf\n",50-numEpochs, train_error, time_taken);

		if (train_error < threshold) {
			fprintf(stdout, "Early Stopping, Training Error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Total Execution Time: %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_FC.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Accuracy: %.2lf%%\n",
		(1 - double(error) / double(test_cnt))* 100.0);
}
