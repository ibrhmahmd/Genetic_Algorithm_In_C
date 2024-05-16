#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <opencv2/opencv.hpp>

// Define image dimensions
#define WIDTH 200
#define HEIGHT 200

using namespace cv;

// to generate random binary image
void generate_random_binary_image(Mat& image) {
for (int i = 0; i < HEIGHT; i++) {
for (int j = 0; j < WIDTH; j++) {
image.at<uchar>(i, j) = rand() % 2 ? 255 : 0;
}
}
}

// to calculate fitness
double calculate_fitness(Mat& image, Mat& target_image) {
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
void select_parents(Mat* population, double *fitness_scores, int num_parents, Mat* selected_parents) {
    double total_fitness = 0.0;
    for (int i = 0; i < HEIGHT; i++) {
        total_fitness += fitness_scores[i];
    }
    for (int i = 0; i < num_parents; i++) {
        double rand_num = (double)rand() / RAND_MAX * total_fitness;
        double sum = 0.0;
        for (int j = 0; j < HEIGHT; j++) {
            sum += fitness_scores[j];
            if (sum >= rand_num) {
                selected_parents[i] = population[j];
                break;
            }
        }
    }
}

// to perform crossover
void crossover(Mat& parent1, Mat& parent2, Mat& offspring) {
int crossover_point = rand() % (WIDTH - 1) + 1;
for (int i = 0; i < WIDTH; i++) {
offspring.at<uchar>(i) = i < crossover_point ? parent1.at<uchar>(i) : parent2.at<uchar>(i);
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
void replace_population(Mat* population, Mat* offspring, double *fitness_scores, int num_offspring) {
    for (int i = 0; i < num_offspring; i++) {
        int least_fit_index = 0;
        for (int j = 1; j < HEIGHT; j++) {
            if (fitness_scores[j] < fitness_scores[least_fit_index]) {
                least_fit_index = j;
            }
        }
        population[least_fit_index] = offspring[i];
        fitness_scores[least_fit_index] = calculate_fitness(offspring[i], target_image);
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL));

    // Load target image
    Mat target_image = imread("path_to_target_image.png", IMREAD_GRAYSCALE);

    // Create window for displaying image
    namedWindow("Best Image", WINDOW_NORMAL);

    // Define variables
    int population_size = 100;
    int num_generations = 100;
    double mutation_rate = 0.01;

    // Initialize population
    Mat population[HEIGHT];
    for (int i = 0; i < HEIGHT; i++) {
        population[i] = Mat(HEIGHT, WIDTH, CV_8UC1);
        generate_random_binary_image(population[i]);
    }

    // Genetic algorithm loop
    for (int generation = 0; generation < num_generations; generation++) {
        // Calculate fitness scores
        double fitness_scores[HEIGHT];
        for (int i = 0; i < HEIGHT; i++) {
            fitness_scores[i] = calculate_fitness(population[i], target_image);
        }

        // Find the index of the best image
        int best_image_index = 0;
        for (int i = 1; i < HEIGHT; i++) {
            if (fitness_scores[i] > fitness_scores[best_image_index]) {
                best_image_index = i;
            }
        }
        Mat best_image = population[best_image_index];

        // Display best image
        imshow("Best Image", best_image);
        waitKey(1);

        // Perform selection, crossover, mutation, and replacement
        Mat selected_parents[2];
        Mat offspring = Mat(HEIGHT, WIDTH, CV_8UC1);
        select_parents(population, fitness_scores, 2, selected_parents);
        crossover(selected_parents[0], selected_parents[1], offspring);
        mutate(offspring, mutation_rate);
        replace_population(population, &offspring, fitness_scores, 1);
    }

    return 0;
}
