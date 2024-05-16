#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Define image dimensions
#define WIDTH 200
#define HEIGHT 200

using namespace cv;
using namespace std;

// to generate random binary image
void generate_random_binary_image(Mat& image) {
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
image.at<uchar>(i, j) = rand() % 2 ? 255 : 0;
}
}
}

// to calculate fitness
double calculate_fitness(const Mat& image, const Mat& target_image) {
double mse = 0.0;
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
double diff = (double)(image.at<uchar>(i, j) - target_image.at<uchar>(i, j));
mse += diff * diff;
}
}
return -mse; // Negative MSE because we want to maximize fitness
}

// to select parents
void select_parents(const vector<Mat>& population, const vector<double>& fitness_scores, int num_parents, vector<Mat>& selected_parents) {
double total_fitness = 0.0;
for (double score : fitness_scores) {
total_fitness += score;
}
for (int i = 0; i < num_parents; i++) {
double rand_num = (double)rand() / RAND_MAX * total_fitness;
double sum = 0.0;
for (size_t j = 0; j < population.size(); j++) {
sum += fitness_scores[j];
if (sum >= rand_num) {
selected_parents[i] = population[j];
break;
}
}
}
}

// to perform crossover
void crossover(const Mat& parent1, const Mat& parent2, Mat& offspring) {
int crossover_point = rand() % (WIDTH - 1) + 1;
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
offspring.at<uchar>(i, j) = j < crossover_point ? parent1.at<uchar>(i, j) : parent2.at<uchar>(i, j);
}
}
}

// to perform mutation
void mutate(Mat& image, double mutation_rate) {
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
if ((double)rand() / RAND_MAX < mutation_rate) {
image.at<uchar>(i, j) = rand() % 256;
}
}
}
}

// to replace population with offspring
void replace_population(vector<Mat>& population, const vector<Mat>& offspring, vector<double>& fitness_scores, const Mat& target_image) {
for (const Mat& child : offspring) {
int least_fit_index = 0;
for (size_t j = 1; j < population.size(); j++) {
if (fitness_scores[j] < fitness_scores[least_fit_index]) {
least_fit_index = j;
}
}
population[least_fit_index] = child;
fitness_scores[least_fit_index] = calculate_fitness(child, target_image);
}
}

int main() {
    // Seed random number generator
    srand(time(NULL));

    // Load target image
    Mat target_image = imread("path_to_target_image.png", IMREAD_GRAYSCALE);
    if (target_image.empty()) {
        cerr << "Error: Could not load target image" << endl;
        return -1;
    }

    // Resize target image to match dimensions
    resize(target_image, target_image, Size(WIDTH, HEIGHT));

    // Create window for displaying image
    namedWindow("Best Image", WINDOW_NORMAL);

    // Define variables
    int population_size = 100;
    int num_generations = 100;
    double mutation_rate = 0.01;

    // Initialize population
    vector<Mat> population(population_size);
    for (int i = 0; i < population_size; i++) {
        population[i] = Mat(HEIGHT, WIDTH, CV_8UC1);
        generate_random_binary_image(population[i]);
    }

    // Genetic algorithm loop
    for (int generation = 0; generation < num_generations; generation++) {
        // Calculate fitness scores
        vector<double> fitness_scores(population_size);
        for (int i = 0; i < population_size; i++) {
            fitness_scores[i] = calculate_fitness(population[i], target_image);
        }

        // Find the index of the best image
        int best_image_index = 0;
        for (int i = 1; i < population_size; i++) {
            if (fitness_scores[i] > fitness_scores[best_image_index]) {
                best_image_index = i;
            }
        }
        Mat best_image = population[best_image_index];

        // Display best image
        imshow("Best Image", best_image);
        waitKey(1);

        // Perform selection, crossover, mutation, and replacement
        vector<Mat> selected_parents(2);
        vector<Mat> offspring(1, Mat(HEIGHT, WIDTH, CV_8UC1));
        select_parents(population, fitness_scores, 2, selected_parents);
        crossover(selected_parents[0], selected_parents[1], offspring[0]);
        mutate(offspring[0], mutation_rate);
        replace_population(population, offspring, fitness_scores, target_image);
    }

    return 0;
}
