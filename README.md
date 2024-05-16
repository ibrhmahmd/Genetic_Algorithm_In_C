# A Aaive Attempt To Write a Genetic Algorithm Image Generator

This project implements a genetic algorithm to generate a binary image that evolves to match a target image. The algorithm uses selection, crossover, mutation, and replacement to iteratively improve the population of images.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of genetic algorithms in image generation and optimization. A population of random binary images is evolved over multiple generations to match a target image as closely as possible. The fitness of each image is determined by comparing it to the target image, and images are selected, crossed over, mutated, and replaced to form new generations.

## Requirements

- C++ compiler (e.g., g++)
- OpenCV library
- pkg-config

## Installation

1. **Clone the repository**:
    ```sh
    git clonehttps://github.com/ibrhmahmd/Genetic_Algorithm_In_C
    cd genetic-algorithm-image-generator
    ```

2. **Install OpenCV**:
    - On Ubuntu:
        ```sh
        sudo apt-get update
        sudo apt-get install libopencv-dev
        ```
    - Alternatively, you can install OpenCV from source. Follow the instructions on the [OpenCV official documentation](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html).

3. **Compile the code**:
    ```sh
    g++ -o genetic_algorithm genetic_algorithm.cpp `pkg-config --cflags --libs opencv4`
    ```

## Usage

1. **Place your target image**:
    - Ensure you have a target image in the same directory or specify the path in the code.

2. **Run the program**:
    ```sh
    ./genetic_algorithm
    ```

3. **Observe the evolution**:
    - The best image of each generation will be displayed in a window titled "Best Image".

## Code Overview

- `genetic_algorithm.cpp`: The main source file containing the implementation of the genetic algorithm.
  - `generate_random_binary_image()`: Generates a random binary image.
  - `calculate_fitness()`: Calculates the fitness of an image by comparing it to the target image.
  - `select_parents()`: Selects parents based on their fitness scores.
  - `crossover()`: Performs crossover between two parent images to create an offspring.
  - `mutate()`: Applies mutation to an image.
  - `replace_population()`: Replaces the least fit individuals in the population with new offspring.
  - `main()`: The main function where the genetic algorithm is executed.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.
