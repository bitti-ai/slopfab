#include "vidfab/sha256.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#endif

namespace vidfab {
namespace {

constexpr std::array<uint32_t, 64> kRound{
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

uint32_t rotate_right(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32u - bits));
}

class Sha256 {
 public:
  void update(const void* data, size_t bytes) {
    if (bytes > (std::numeric_limits<uint64_t>::max() - total_bytes_))
      throw std::overflow_error("SHA-256 input length overflow");
    total_bytes_ += bytes;
    const auto* cursor = static_cast<const uint8_t*>(data);
    if (buffered_ != 0) {
      const size_t copy = std::min(bytes, block_.size() - buffered_);
      std::memcpy(block_.data() + buffered_, cursor, copy);
      buffered_ += copy;
      cursor += copy;
      bytes -= copy;
      if (buffered_ == block_.size()) {
        transform(block_.data());
        buffered_ = 0;
      }
    }
    while (bytes >= block_.size()) {
      transform(cursor);
      cursor += block_.size();
      bytes -= block_.size();
    }
    if (bytes != 0) {
      std::memcpy(block_.data(), cursor, bytes);
      buffered_ = bytes;
    }
  }

  Sha256Digest finish() {
    if (total_bytes_ > std::numeric_limits<uint64_t>::max() / 8u)
      throw std::overflow_error("SHA-256 bit length overflow");
    const uint64_t bit_length = total_bytes_ * 8u;
    block_[buffered_++] = 0x80u;
    if (buffered_ > 56) {
      std::fill(block_.begin() + buffered_, block_.end(), uint8_t{0});
      transform(block_.data());
      buffered_ = 0;
    }
    std::fill(block_.begin() + buffered_, block_.begin() + 56, uint8_t{0});
    for (unsigned i = 0; i < 8; ++i)
      block_[63 - i] = static_cast<uint8_t>(bit_length >> (8u * i));
    transform(block_.data());
    Sha256Digest result{};
    for (size_t i = 0; i < state_.size(); ++i) {
      result[i * 4] = static_cast<uint8_t>(state_[i] >> 24);
      result[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
      result[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
      result[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
    }
    return result;
  }

 private:
  void transform(const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (size_t i = 0; i < 16; ++i) {
      words[i] = (uint32_t(block[i * 4]) << 24) |
                 (uint32_t(block[i * 4 + 1]) << 16) |
                 (uint32_t(block[i * 4 + 2]) << 8) |
                 uint32_t(block[i * 4 + 3]);
    }
    for (size_t i = 16; i < words.size(); ++i) {
      const uint32_t s0 = rotate_right(words[i - 15], 7) ^
                          rotate_right(words[i - 15], 18) ^
                          (words[i - 15] >> 3);
      const uint32_t s1 = rotate_right(words[i - 2], 17) ^
                          rotate_right(words[i - 2], 19) ^
                          (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
    uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (size_t i = 0; i < words.size(); ++i) {
      const uint32_t sum1 = rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + sum1 + choice + kRound[i] + words[i];
      const uint32_t sum0 = rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sum0 + majority;
      h=g;g=f;f=e;e=d+temp1;d=c;c=b;b=a;a=temp1+temp2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
    state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }

  std::array<uint32_t, 8> state_{
      0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
      0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  std::array<uint8_t, 64> block_{};
  size_t buffered_ = 0;
  uint64_t total_bytes_ = 0;
};

}  // namespace

Sha256Digest sha256_bytes(const void* data, size_t bytes) {
  if (!data && bytes != 0) throw std::invalid_argument("SHA-256 null input");
  Sha256 hash;
  hash.update(data, bytes);
  return hash.finish();
}

Sha256Digest sha256_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("SHA-256 cannot open " + path);
  std::vector<uint8_t> chunk(4u << 20);
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  auto fail = [&](const char* message) -> void {
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error(message + std::string(" for ") + path);
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                  nullptr, 0) < 0 ||
      BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
    fail("SHA-256 provider failed");
  }
  while (input) {
    input.read(reinterpret_cast<char*>(chunk.data()),
               static_cast<std::streamsize>(chunk.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 && BCryptHashData(
            hash, chunk.data(), static_cast<ULONG>(count), 0) < 0) {
      fail("SHA-256 update failed");
    }
  }
  if (!input.eof()) fail("SHA-256 read failed");
  Sha256Digest result{};
  if (BCryptFinishHash(hash, result.data(),
                       static_cast<ULONG>(result.size()), 0) < 0) {
    fail("SHA-256 finish failed");
  }
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  return result;
#else
  Sha256 hash;
  while (input) {
    input.read(reinterpret_cast<char*>(chunk.data()),
               static_cast<std::streamsize>(chunk.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) hash.update(chunk.data(), static_cast<size_t>(count));
  }
  if (!input.eof()) throw std::runtime_error("SHA-256 read failed for " + path);
  return hash.finish();
#endif
}

}  // namespace vidfab
