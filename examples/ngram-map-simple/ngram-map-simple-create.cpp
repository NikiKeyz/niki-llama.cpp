#include "common.h"
#include "ngram-map-simple.h"
#include "log.h"

#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cinttypes>

static void print_usage(int /*argc*/, char ** argv) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m, --model MODEL     model path\n");
    fprintf(stderr, "  -f, --file FILE       input file(s) to process (can be specified multiple times)\n");
    fprintf(stderr, "  -o, --output FILE     output cache file\n");
    fprintf(stderr, "  -n, --ngram N         n-gram size for key (default: %d)\n", 8);
    fprintf(stderr, "  -M, --mgram M         m-gram size for value (default: %d)\n", 16);
    fprintf(stderr, "  -c, --chunk N         chunk size for processing (default: %d)\n", 10000);
    fprintf(stderr, "\n");
    fprintf(stderr, "example:\n");
    fprintf(stderr, "  %s -m /path/to/model -f /path/to/code -o cache.bin\n", argv[0]);
    fprintf(stderr, "\n");
}

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open file " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    // Parse arguments
    std::string model_path;
    std::vector<std::string> input_files;
    std::string output_file;
    uint16_t ngram_size = 8;
    uint16_t mgram_size = 16;
    int chunk_size = 10000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            model_path = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            input_files.push_back(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-n" || arg == "--ngram") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            ngram_size = std::stoi(argv[i]);
        } else if (arg == "-M" || arg == "--mgram") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            mgram_size = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--chunk") {
            if (++i >= argc) {
                fprintf(stderr, "error: missing argument for %s\n", argv[i - 1]);
                return 1;
            }
            chunk_size = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "error: unknown option %s\n", argv[i]);
            print_usage(argc, argv);
            return 1;
        }
    }

    // Validate arguments
    if (model_path.empty()) {
        fprintf(stderr, "error: model path is required\n");
        print_usage(argc, argv);
        return 1;
    }
    if (input_files.empty()) {
        fprintf(stderr, "error: at least one input file is required\n");
        print_usage(argc, argv);
        return 1;
    }
    if (output_file.empty()) {
        fprintf(stderr, "error: output file is required\n");
        print_usage(argc, argv);
        return 1;
    }

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;

    auto model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", model_path.c_str());
        return 1;
    }

    auto vocab = llama_model_get_vocab(model);

    // Create ngram map
    auto * map = common_ngram_map_simple_create(ngram_size, mgram_size);

    // Process input files
    int64_t total_tokens = 0;
    for (const auto & file_path : input_files) {
        LOG_INF("processing file: %s\n", file_path.c_str());

        std::string content = read_file(file_path);

        // Tokenize content with buffer reallocation if needed
        std::vector<llama_token> tokens;
        tokens.reserve(content.size() * 2);
        int n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            // Buffer too small, reallocate
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, content.c_str(), content.size(), tokens.data(), tokens.size(), true, true);
        }
        if (n_tokens < 0) {
            fprintf(stderr, "error: failed to tokenize file %s\n", file_path.c_str());
            common_ngram_map_simple_free(map);
            llama_model_free(model);
            return 1;
        }
        tokens.resize(n_tokens);

        // Process in chunks
        for (int64_t i = 0; i < n_tokens; i += chunk_size) {
            int64_t end = std::min(i + chunk_size, (int64_t)n_tokens);
            llama_tokens chunk(tokens.begin() + i, tokens.begin() + end);
            common_ngram_map_simple_add(map, chunk);
        }

        total_tokens += n_tokens;
        LOG_INF("processed %d tokens from %s\n", n_tokens, file_path.c_str());
    }

    // Save to output file
    common_ngram_map_simple_save(map, output_file);

    // Print statistics
    size_t n_entries = common_ngram_map_simple_size(map);
    LOG_INF("total tokens processed: %" PRId64 "\n", total_tokens);
    LOG_INF("total entries in map: %zu\n", n_entries);

    // Cleanup
    common_ngram_map_simple_free(map);
    llama_model_free(model);

    return 0;
}
