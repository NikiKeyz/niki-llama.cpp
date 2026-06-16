#pragma once
//
// common/ngram-map-simple.h: n-gram map with frequency tracking for speculative decoding
//
// Similar to ngram-simple but stores frequency counts for each (key, value) pair.
// Returns the most frequent m-gram continuation when a matching n-gram is found.
// Supports serialization to/from binary files for persistent storage across sessions.
//

#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

// Maximum number of m-gram values stored for each key n-gram
#define COMMON_NGRAM_MAP_SIMPLE_MAX_VALUES 4

// Magic number for binary file format validation
#define COMMON_NGRAM_MAP_SIMPLE_MAGIC 0x4e47524d  // "NGRM"

// Version of the binary file format
#define COMMON_NGRAM_MAP_SIMPLE_VERSION 1

// Hash function for n-gram keys
struct common_ngram_map_simple_key_hash {
    size_t operator()(const std::vector<llama_token> & key) const {
        // Use Fibonacci hashing for better distribution
        size_t hash = 0;
        for (llama_token t : key) {
            hash ^= static_cast<size_t>(t) * 11400714819323198485llu;
        }
        return hash;
    }
};

// Value m-gram with its occurrence count
struct common_ngram_map_simple_value {
    llama_tokens tokens;
    uint16_t count;
};

// Entry in the ngram map storing a key n-gram and its value m-grams
struct common_ngram_map_simple_entry {
    // Number of times this key was seen
    uint32_t key_num;

    // Number of distinct value m-grams stored
    uint8_t n_values;

    // Padding for alignment
    uint8_t _pad;

    // Value m-grams and their occurrence counts
    std::vector<common_ngram_map_simple_value> values;
};

// Configuration for ngram map simple
struct common_ngram_map_simple_config {
    uint16_t size_key;    // size of key n-grams
    uint16_t size_value;  // size of value m-grams
};

// Opaque type for the ngram map
struct common_ngram_map_simple {
    uint16_t size_key;
    uint16_t size_value;

    // Map from key n-gram to entry with value m-grams and counts
    std::unordered_map<std::vector<llama_token>, common_ngram_map_simple_entry,
                       common_ngram_map_simple_key_hash> entries;
};

// Create a new ngram map simple with the given configuration
common_ngram_map_simple * common_ngram_map_simple_create(
    uint16_t size_key,
    uint16_t size_value);

// Free the ngram map simple
void common_ngram_map_simple_free(common_ngram_map_simple * map);

// Add tokens to the ngram map
// This processes a sliding window over the tokens and updates key-value counts
void common_ngram_map_simple_add(
    common_ngram_map_simple * map,
    const llama_tokens & tokens);

// Generate a draft from the ngram map
// Returns the most frequent m-gram continuation for the current key n-gram
// Returns empty vector if no match found or if key_num is below min_hits
llama_tokens common_ngram_map_simple_draft(
    const common_ngram_map_simple * map,
    const llama_tokens & tokens,
    llama_token sampled,
    uint16_t min_hits);

// Save the ngram map to a binary file
void common_ngram_map_simple_save(
    const common_ngram_map_simple * map,
    const std::string & filename);

// Load the ngram map from a binary file
common_ngram_map_simple * common_ngram_map_simple_load(
    const std::string & filename);

// Merge another ngram map into this one
// Counts are added together for matching (key, value) pairs
void common_ngram_map_simple_merge(
    common_ngram_map_simple * target,
    const common_ngram_map_simple * source);

// Get the number of entries in the map
size_t common_ngram_map_simple_size(const common_ngram_map_simple * map);
