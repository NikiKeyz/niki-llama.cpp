#include "ngram-map-simple.h"
#include "log.h"

#include <fstream>
#include <algorithm>
#include <cinttypes>

common_ngram_map_simple * common_ngram_map_simple_create(
    uint16_t size_key,
    uint16_t size_value) {
    auto * map = new common_ngram_map_simple();
    map->size_key = size_key;
    map->size_value = size_value;
    return map;
}

void common_ngram_map_simple_free(common_ngram_map_simple * map) {
    delete map;
}

void common_ngram_map_simple_add(
    common_ngram_map_simple * map,
    const llama_tokens & tokens) {
    const size_t n = tokens.size();
    const uint16_t key_size = map->size_key;
    const uint16_t value_size = map->size_value;

    // Need at least key + value tokens
    if (n < key_size + value_size) {
        return;
    }

    for (size_t i = 0; i <= n - key_size - value_size; ++i) {
        // Extract key n-gram
        std::vector<llama_token> key(key_size);
        for (uint16_t k = 0; k < key_size; ++k) {
            key[k] = tokens[i + k];
        }

        // Extract value m-gram
        llama_tokens value(value_size);
        for (uint16_t v = 0; v < value_size; ++v) {
            value[v] = tokens[i + key_size + v];
        }

        // Find or create entry
        auto it = map->entries.find(key);
        if (it == map->entries.end()) {
            common_ngram_map_simple_entry entry;
            entry.key_num = 1;
            entry.n_values = 1;
            entry.values.resize(1);
            entry.values[0].tokens = value;
            entry.values[0].count = 1;
            map->entries.emplace(key, entry);
        } else {
            common_ngram_map_simple_entry & entry = it->second;
            entry.key_num++;

            // Check if this value already exists
            bool found = false;
            for (uint8_t v = 0; v < entry.n_values; ++v) {
                if (entry.values[v].tokens == value) {
                    entry.values[v].count++;
                    found = true;
                    break;
                }
            }

            // Add new value if not found and space available
            if (!found && entry.n_values < COMMON_NGRAM_MAP_SIMPLE_MAX_VALUES) {
                entry.values.emplace_back();
                entry.values.back().tokens = value;
                entry.values.back().count = 1;
                entry.n_values++;
            }
        }
    }
}

llama_tokens common_ngram_map_simple_draft(
    const common_ngram_map_simple * map,
    const llama_tokens & tokens,
    llama_token sampled,
    uint16_t min_hits) {
    if (!map) {
        return {};
    }

    const size_t n = tokens.size();
    const uint16_t key_size = map->size_key;

    // Need at least key_size tokens
    if (n < key_size) {
        return {};
    }

    // Build key from last key_size tokens + sampled
    std::vector<llama_token> key(key_size);
    for (uint16_t k = 0; k < key_size - 1; ++k) {
        key[k] = tokens[n - key_size + 1 + k];
    }
    key[key_size - 1] = sampled;

    // Look up key in map
    auto it = map->entries.find(key);
    if (it == map->entries.end()) {
        return {};
    }

    const common_ngram_map_simple_entry & entry = it->second;

    // Check minimum hits
    if (entry.key_num < min_hits) {
        return {};
    }

    // Find the most frequent value
    uint16_t max_count = 0;
    uint8_t best_value_idx = 0;
    for (uint8_t v = 0; v < entry.n_values; ++v) {
        if (entry.values[v].count > max_count) {
            max_count = entry.values[v].count;
            best_value_idx = v;
        }
    }

    // Return the best value m-gram
    return entry.values[best_value_idx].tokens;
}

void common_ngram_map_simple_save(
    const common_ngram_map_simple * map,
    const std::string & filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: failed to open file %s\n", __func__, filename.c_str());
        return;
    }

    // Write header
    uint32_t magic = COMMON_NGRAM_MAP_SIMPLE_MAGIC;
    uint32_t version = COMMON_NGRAM_MAP_SIMPLE_VERSION;
    uint16_t size_key = map->size_key;
    uint16_t size_value = map->size_value;
    uint32_t n_entries = static_cast<uint32_t>(map->entries.size());

    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&size_key), sizeof(size_key));
    file.write(reinterpret_cast<const char *>(&size_value), sizeof(size_value));
    file.write(reinterpret_cast<const char *>(&n_entries), sizeof(n_entries));

    // Write entries
    for (const auto & [key, entry] : map->entries) {
        // Write key
        uint16_t key_size = static_cast<uint16_t>(key.size());
        file.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
        for (llama_token t : key) {
            file.write(reinterpret_cast<const char *>(&t), sizeof(t));
        }

        // Write entry data
        file.write(reinterpret_cast<const char *>(&entry.key_num), sizeof(entry.key_num));
        file.write(reinterpret_cast<const char *>(&entry.n_values), sizeof(entry.n_values));

        // Write values
        for (uint8_t v = 0; v < entry.n_values; ++v) {
            const auto & value = entry.values[v];
            file.write(reinterpret_cast<const char *>(&value.count), sizeof(value.count));
            uint16_t value_size = static_cast<uint16_t>(value.tokens.size());
            file.write(reinterpret_cast<const char *>(&value_size), sizeof(value_size));
            for (llama_token t : value.tokens) {
                file.write(reinterpret_cast<const char *>(&t), sizeof(t));
            }
        }
    }

    file.close();
    LOG_INF("%s: saved %u entries to %s\n", __func__, n_entries, filename.c_str());
}

common_ngram_map_simple * common_ngram_map_simple_load(
    const std::string & filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: failed to open file %s\n", __func__, filename.c_str());
        return nullptr;
    }

    // Read header
    uint32_t magic, version;
    uint16_t size_key, size_value;
    uint32_t n_entries;

    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&size_key), sizeof(size_key));
    file.read(reinterpret_cast<char *>(&size_value), sizeof(size_value));
    file.read(reinterpret_cast<char *>(&n_entries), sizeof(n_entries));

    // Validate magic and version
    if (magic != COMMON_NGRAM_MAP_SIMPLE_MAGIC) {
        LOG_ERR("%s: invalid magic number in file %s\n", __func__, filename.c_str());
        return nullptr;
    }
    if (version != COMMON_NGRAM_MAP_SIMPLE_VERSION) {
        LOG_ERR("%s: unsupported version %u in file %s\n", __func__, version, filename.c_str());
        return nullptr;
    }
    if (n_entries > 10000000) {  // 10M maximum
        LOG_ERR("%s: too many entries (%u) in file %s\n", __func__, n_entries, filename.c_str());
        return nullptr;
    }

    // Create map
    auto * map = common_ngram_map_simple_create(size_key, size_value);

    // Read entries
    for (uint32_t i = 0; i < n_entries; ++i) {
        // Read key
        uint16_t key_size;
        file.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
        std::vector<llama_token> key(key_size);
        for (uint16_t k = 0; k < key_size; ++k) {
            file.read(reinterpret_cast<char *>(&key[k]), sizeof(key[k]));
        }

        // Read entry data
        common_ngram_map_simple_entry entry;
        file.read(reinterpret_cast<char *>(&entry.key_num), sizeof(entry.key_num));
        file.read(reinterpret_cast<char *>(&entry.n_values), sizeof(entry.n_values));

        // Read values
        entry.values.resize(entry.n_values);
        for (uint8_t v = 0; v < entry.n_values; ++v) {
            file.read(reinterpret_cast<char *>(&entry.values[v].count), sizeof(entry.values[v].count));
            uint16_t value_size;
            file.read(reinterpret_cast<char *>(&value_size), sizeof(value_size));
            entry.values[v].tokens.resize(value_size);
            for (uint16_t k = 0; k < value_size; ++k) {
                file.read(reinterpret_cast<char *>(&entry.values[v].tokens[k]), sizeof(entry.values[v].tokens[k]));
            }
        }

        map->entries.emplace(key, entry);
    }

    file.close();
    LOG_INF("%s: loaded %u entries from %s\n", __func__, n_entries, filename.c_str());
    return map;
}

void common_ngram_map_simple_merge(
    common_ngram_map_simple * target,
    const common_ngram_map_simple * source) {
    for (const auto & [key, source_entry] : source->entries) {
        auto it = target->entries.find(key);
        if (it == target->entries.end()) {
            // Add new entry
            target->entries.emplace(key, source_entry);
        } else {
            // Merge counts
            common_ngram_map_simple_entry & target_entry = it->second;
            target_entry.key_num += source_entry.key_num;

            // Merge value counts
            for (uint8_t v = 0; v < source_entry.n_values; ++v) {
                // Find matching value in target
                bool found = false;
                for (uint8_t t = 0; t < target_entry.n_values; ++t) {
                    if (target_entry.values[t].tokens == source_entry.values[v].tokens) {
                        target_entry.values[t].count += source_entry.values[v].count;
                        found = true;
                        break;
                    }
                }

                // Add new value if not found and space available
                if (!found && target_entry.n_values < COMMON_NGRAM_MAP_SIMPLE_MAX_VALUES) {
                    target_entry.values.emplace_back();
                    target_entry.values.back().tokens = source_entry.values[v].tokens;
                    target_entry.values.back().count = source_entry.values[v].count;
                    target_entry.n_values++;
                }
            }
        }
    }
}

size_t common_ngram_map_simple_size(const common_ngram_map_simple * map) {
    return map->entries.size();
}
