/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define DEBUG_OUTPUT  // Comment out to remove debug output

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	#ifdef DEBUG_OUTPUT
	cout << "~~~~~~~~~~ init ~~~~~~~~~~" << endl;
	#endif

	// Set number of particles
	num_particles = 10;

	#ifdef DEBUG_OUTPUT
	cout << "Num particles: " << num_particles << endl;
	#endif

	// Generate random distributions
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> nd_x(0.0, std[0]);
	normal_distribution<double> nd_y(0.0, std[1]);
	normal_distribution<double> nd_theta(0.0, std[2]);
	
	// Init weights
	weights.resize(num_particles);
	fill(weights.begin(), weights.end(), 1.0);

	// Init particles
	for (int i=0 ; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = x + nd_x(gen);
		p.y = y + nd_y(gen);
		p.theta = theta + nd_theta(gen);
		p.weight = weights[i];
		particles.push_back(p);
	}

	// Finish initializing
	is_initialized = true;

	#ifdef DEBUG_OUTPUT
	cout << "Particles initialized" << endl;
	#endif
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	#ifdef DEBUG_OUTPUT
	cout << "~~~~~~~~~~ prediction ~~~~~~~~~~" << endl;
	#endif

	// Generate random distributions
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<double> nd_x(0.0, std_pos[0]);
	normal_distribution<double> nd_y(0.0, std_pos[1]);
	normal_distribution<double> nd_theta(0.0, std_pos[2]);

	double travel_scale = 1.0;
	if (yaw_rate != 0.0) {
		travel_scale = velocity / yaw_rate; // scaling factor for distance
	}

	// Predict new particle locations
	for (int i = 0; i < num_particles; ++i) {
		#ifdef DEBUG_OUTPUT
		printf("Particle %d: (%f, %f, %f)->", particles[i].id, particles[i].x, particles[i].y, particles[i].theta);
		#endif

		double travel_dist = 0; // either yaw dist or forward dist
		if (yaw_rate != 0.0) {
			travel_dist = yaw_rate * delta_t;
			particles[i].x += travel_scale * (sin(particles[i].theta + travel_dist) - sin(particles[i].theta)) + nd_x(gen);
			particles[i].y += travel_scale * (cos(particles[i].theta) - cos(particles[i].theta + travel_dist)) + nd_y(gen);
			particles[i].theta += travel_dist + nd_theta(gen);
		} else {
			travel_dist = velocity * delta_t;
			particles[i].x += travel_dist * cos(particles[i].theta) + nd_x(gen);
			particles[i].y += travel_dist * sin(particles[i].theta) + nd_y(gen);
			particles[i].theta += nd_theta(gen);
		}
		particles[i].theta = fmod(particles[i].theta, 2 * M_PI);

		#ifdef DEBUG_OUTPUT
		printf("(%f, %f, %f)\n", particles[i].x, particles[i].y, particles[i].theta);
		#endif
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i) {
		double curr_dist = 0;
		double closest_dist = numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); ++j) {
			curr_dist = abs(dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y));
			if (curr_dist < closest_dist) {
				observations[i].id = predicted[j].id;
				closest_dist = curr_dist;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	#ifdef DEBUG_OUTPUT
	cout << "~~~~~~~~~~ updateWeights ~~~~~~~~~~" << endl;
	#endif

	for (int i = 0; i < num_particles; ++i) {
		// Predict sensor measurements for each particle within sensor range
		vector<LandmarkObs> predicted_locations;
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			if (abs(dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f,
			  particles[i].x, particles[i].y)) <= sensor_range) {
				LandmarkObs l;
				l.id = map_landmarks.landmark_list[j].id_i;
				l.x = map_landmarks.landmark_list[j].x_f;
				l.y = map_landmarks.landmark_list[j].y_f;
				predicted_locations.push_back(l);
			}
		}

		// Transform sensor measurements to MAP coordinate system
		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs l;
			l.id = 0;
			l.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			l.y = particles[i].y + sin(particles[i].theta) * observations[j].y + cos(particles[i].theta) * observations[j].x;
			transformed_observations.push_back(l);
		}

		// Find associations between observations and map landmarks
		dataAssociation(predicted_locations, transformed_observations);
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		for (int j = 0; j > transformed_observations.size(); ++j) {
			particles[i].associations.push_back(transformed_observations[j].id);
			particles[i].sense_x.push_back(transformed_observations[j].x);
			particles[i].sense_y.push_back(transformed_observations[j].y);
		}

		// Update weight based on associations
		// FIXME: All weights computed to be 0
		double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		particles[i].weight = 1;
		for (int j = 0; j < transformed_observations.size(); ++j) {
			double x_obs = transformed_observations[j].x;
			double y_obs = transformed_observations[j].y;
			double mu_x = map_landmarks.landmark_list[transformed_observations[j].id-1].x_f;
			double mu_y = map_landmarks.landmark_list[transformed_observations[j].id-1].y_f;
			double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(std_landmark[1], 2)));
			particles[i].weight *= gauss_norm * exp(-exponent);  // Some of the particle weights come out as 0
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// Create vector of particle weights

	#ifdef DEBUG_OUTPUT
	cout << "~~~~~~~~~~ resample ~~~~~~~~~~" << endl;
	#endif

	weights.clear();
	for (int i = 0; i < num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}

	// Create sampler
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> d(weights.begin(), weights.end());

	// Resample particles
	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[d(gen)];
		p.id = i + 1;
		resampled_particles.push_back(p);
	}
	particles = resampled_particles;

	#ifdef DEBUG_OUTPUT
	cout << "Finished resampling" << endl;
	#endif
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
