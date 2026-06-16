#include "ngram-map-simple.h"
#include <cassert>
#include <cstdio>
#include <vector>

static void test_create_free() {
    printf("test_create_free...\n");
    auto * map = common_ngram_map_simple_create(4, 8);
    assert(map != nullptr);
    assert(map->size_key == 4);
    assert(map->size_value == 8);
    common_ngram_map_simple_free(map);
    printf("  PASSED\n");
}

static void test_add_draft() {
    printf("test_add_draft...\n");
    auto * map = common_ngram_map_simple_create(4, 8);

    // Create a test token sequence
    llama_tokens tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // Add tokens to the map
    common_ngram_map_simple_add(map, tokens);

    // Verify that the map has entries
    assert(common_ngram_map_simple_size(map) > 0);

    // Try to draft with a matching key
    llama_tokens prompt = {1, 2, 3, 4};
    llama_token sampled = 5;

    llama_tokens draft = common_ngram_map_simple_draft(map, prompt, sampled, 1);

    // Verify that the draft contains the expected tokens
    assert(draft.size() == 8);
    assert(draft[0] == 6);
    assert(draft[1] == 7);
    assert(draft[2] == 8);
    assert(draft[3] == 9);
    assert(draft[4] == 10);
    assert(draft[5] == 11);
    assert(draft[6] == 12);
    assert(draft[7] == 13);

    common_ngram_map_simple_free(map);
    printf("  PASSED\n");
}

static void test_save_load() {
    printf("test_save_load...\n");
    auto * map = common_ngram_map_simple_create(4, 8);

    // Create a test token sequence
    llama_tokens tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // Add tokens to the map
    common_ngram_map_simple_add(map, tokens);

    // Save to file
    common_ngram_map_simple_save(map, "/tmp/test_ngram_map_simple.bin");

    // Free the original map
    common_ngram_map_simple_free(map);

    // Load from file
    auto * loaded_map = common_ngram_map_simple_load("/tmp/test_ngram_map_simple.bin");
    assert(loaded_map != nullptr);

    // Verify that the loaded map has the same entries
    assert(common_ngram_map_simple_size(loaded_map) == common_ngram_map_simple_size(map));

    // Try to draft with a matching key
    llama_tokens prompt = {1, 2, 3, 4};
    llama_token sampled = 5;

    llama_tokens draft = common_ngram_map_simple_draft(loaded_map, prompt, sampled, 1);

    // Verify that the draft contains the expected tokens
    assert(draft.size() == 8);
    assert(draft[0] == 6);
    assert(draft[1] == 7);
    assert(draft[2] == 8);
    assert(draft[3] == 9);
    assert(draft[4] == 10);
    assert(draft[5] == 11);
    assert(draft[6] == 12);
    assert(draft[7] == 13);

    common_ngram_map_simple_free(loaded_map);
    printf("  PASSED\n");
}

static void test_frequency() {
    printf("test_frequency...\n");
    auto * map = common_ngram_map_simple_create(4, 8);

    // Add the same sequence multiple times
    llama_tokens tokens1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    llama_tokens tokens2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    llama_tokens tokens3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    common_ngram_map_simple_add(map, tokens1);
    common_ngram_map_simple_add(map, tokens2);
    common_ngram_map_simple_add(map, tokens3);

    // Add a different continuation once
    llama_tokens tokens4 = {1, 2, 3, 4, 5, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90};
    common_ngram_map_simple_add(map, tokens4);

    // Try to draft with a matching key
    llama_tokens prompt = {1, 2, 3, 4};
    llama_token sampled = 5;

    llama_tokens draft = common_ngram_map_simple_draft(map, prompt, sampled, 1);

    // Verify that the draft contains the most frequent continuation
    assert(draft.size() == 8);
    assert(draft[0] == 6);  // Most frequent continuation
    assert(draft[1] == 7);
    assert(draft[2] == 8);
    assert(draft[3] == 9);
    assert(draft[4] == 10);
    assert(draft[5] == 11);
    assert(draft[6] == 12);
    assert(draft[7] == 13);

    common_ngram_map_simple_free(map);
    printf("  PASSED\n");
}

int main() {
    printf("Running ngram-map-simple unit tests...\n\n");

    test_create_free();
    test_add_draft();
    test_save_load();
    test_frequency();

    printf("\nAll tests passed!\n");
    return 0;
}
