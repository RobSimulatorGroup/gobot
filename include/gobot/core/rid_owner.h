/* The gobot is a robot simulation platform.
 * Copyright(c) 2021-2022, RobSimulatorGroup, Qiqi Wu<1258552199@qq.com>.
 * Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
 * This version of the GNU Lesser General Public License incorporates the terms and conditions of version 3 of the GNU General Public License.
 * This file is created by Zikun Yu, 23-3-20
*/

#pragma once

#include <atomic>
#include "gobot/core/rid.h"
#include "gobot/core/spin_lock.hpp"
#include "gobot/error_macros.hpp"

namespace gobot {

class RID_AllocBase {
    static std::atomic<uint64_t> base_id_;

protected:
    static RID MakeFromID(uint64_t id) {
        RID rid;
        rid.id_ = id;
        return rid;
    }

    static uint64_t GenID() {
        return base_id_.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    static RID GenRID() {
        return MakeFromID(GenID());
    }

public:
    virtual ~RID_AllocBase() = default;
};

/**
 * @brief Manage memory allocations to a template class T. Upon an RID generated, a new chunk of data will be
 *  allocated with a validator chunk and a free_list chunk for validating ID (and initialization) and recording
 *  free index to the memory chunk separately. In this way, a 64-bit RID should be composed of an upper 32-bit ID
 *  and a lower 32-bit memory chunk index.
 *
 *  Note that Allocation and Initialization are processed separately, where allocation grows memory chunks and
 *  initialize validator and free list while initialization set a flag to indicate the memory is initialized by
 *  some type T. To free an object of type T, not only memory chunk will be freed, validators and free list should
 *  also be initialized for the next allocation.
 */
template <class T, bool THREAD_SAFE = false>
class RID_Alloc : public RID_AllocBase {
public:
    RID MakeRID() {
        RID rid = AllocateRID();
        InitializeRID(rid);
        return rid;
    }

    RID MakeRID(const T &value) {
        RID rid = AllocateRID();
        InitializeRID(rid, value);
        return rid;
    }

    FORCE_INLINE T *GetOrNull(const RID &rid, bool initialize = false) {
        if (rid == RID()) {
            return nullptr;
        }

        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        uint64_t id = rid.GetID();
        uint32_t idx = id & 0xFFFFFFFF;
        if (idx >= max_alloc_) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            return nullptr;
        }

        uint32_t idx_chunk = idx / elements_in_chunk_;
        uint32_t idx_element = idx % elements_in_chunk_;

        uint32_t validator = id >> 32;

        if (initialize) [[unlikely]] {
            if (!(validator_chunks_[idx_chunk][idx_element] & 0x80000000)) [[unlikely]] {
                if (THREAD_SAFE) {
                    spin_lock_.unlock();
                }
                ERR_FAIL_V_MSG(nullptr, "Initializing already initialized RID")
            }

            if ((validator_chunks_[idx_chunk][idx_element] & 0x7FFFFFFF) != validator) [[unlikely]] {
                if (THREAD_SAFE) {
                    spin_lock_.unlock();
                }
                ERR_FAIL_V_MSG(nullptr, "Attempting to initialize the wrong RID")
            }

            validator_chunks_[idx_chunk][idx_element] &= 0x7FFFFFFF; // initialized

        } else if (validator_chunks_[idx_chunk][idx_element] != validator) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            if ((validator_chunks_[idx_chunk][idx_element] & 0x80000000) &&
                validator_chunks_[idx_chunk][idx_element] != 0xFFFFFFFF) {
                ERR_FAIL_V_MSG(nullptr, "Attempting to use an uninitialized RID")
            }
            return nullptr;
        }

        T *ptr = &chunks_[idx_chunk][idx_element];

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }

        return ptr;
    }

    FORCE_INLINE RID AllocateRID() {
        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        if (alloc_count_ == max_alloc_) {
            // allocate a new chunk
            uint32_t chunk_count = alloc_count_ == 0 ? 0 : (max_alloc_ / elements_in_chunk_);

            // grow chunks
            chunks_ = (T **)memrealloc(chunks_, sizeof(T *) * (chunk_count + 1));
            chunks_[chunk_count] = (T *)malloc(sizeof(T) * elements_in_chunk_); //but don't initialize

            // grow validators
            validator_chunks_ = (uint32_t **)realloc(validator_chunks_, sizeof(uint32_t *) * (chunk_count + 1));
            validator_chunks_[chunk_count] = (uint32_t *)malloc(sizeof(uint32_t) * elements_in_chunk_);

            // grow free lists
            free_list_chunks_ = (uint32_t **)realloc(free_list_chunks_, sizeof(uint32_t *) * (chunk_count + 1));
            free_list_chunks_[chunk_count] = (uint32_t *)malloc(sizeof(uint32_t) * elements_in_chunk_);

            // initialize
            for (uint32_t i = 0; i < elements_in_chunk_; i++) {
                // Don't initialize chunk.
                validator_chunks_[chunk_count][i] = 0xFFFFFFFF;
                free_list_chunks_[chunk_count][i] = alloc_count_ + i;
            }

            max_alloc_ += elements_in_chunk_;
        }

        uint32_t free_index = free_list_chunks_[alloc_count_ / elements_in_chunk_][alloc_count_ % elements_in_chunk_];

        uint32_t free_chunk = free_index / elements_in_chunk_;
        uint32_t free_element = free_index % elements_in_chunk_;

        uint32_t validator = GenID() & 0x7FFFFFFF;
        uint64_t id = validator;
        id <<= 32;
        id |= free_index;

        validator_chunks_[free_chunk][free_element] = validator;

        validator_chunks_[free_chunk][free_element] |= 0x80000000; //mark uninitialized bit

        alloc_count_ ++;

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }

        return MakeFromID(id);
    }

    void InitializeRID(RID rid) {
        T *mem = GetOrNull(rid, true);
        ERR_FAIL_COND(!mem);
        new (mem) T; // construct a 'T' object and place it directly into pre-allocated storage at memory address 'mem'
    }

    void InitializeRID(RID rid, const T &value) {
        T *mem = GetOrNull(rid, true);
        ERR_FAIL_COND(!mem);
        new (mem) T(value);
    }

    FORCE_INLINE bool Owns(const RID &rid) const {
        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        uint64_t id = rid.GetID();
        uint32_t idx = id & 0xFFFFFFFF;
        if (idx >= max_alloc_) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            return false;
        }

        uint32_t idx_chunk = idx / elements_in_chunk_;
        uint32_t idx_element = idx % elements_in_chunk_;

        uint32_t validator = id >> 32;

        bool owned = (validator_chunks_[idx_chunk][idx_element] & 0x7FFFFFFF) == validator;

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }

        return owned;
    }

    FORCE_INLINE void Free(const RID &rid) {
        if (THREAD_SAFE) {
            spin_lock_.lock();
        }

        uint64_t id = rid.GetID();
        uint32_t idx = id & 0xFFFFFFFF;
        if (idx >= max_alloc_) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            ERR_FAIL()
        }

        uint32_t idx_chunk = idx / elements_in_chunk_;
        uint32_t idx_element = idx % elements_in_chunk_;

        uint32_t validator = id >> 32;
        if (validator_chunks_[idx_chunk][idx_element] & 0x80000000) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            ERR_FAIL_MSG("Attempted to free an uninitialized or invalid RID")
        } else if (validator_chunks_[idx_chunk][idx_element] != validator) [[unlikely]] {
            if (THREAD_SAFE) {
                spin_lock_.unlock();
            }
            ERR_FAIL()
        }

        chunks_[idx_chunk][idx_element].~T();
        validator_chunks_[idx_chunk][idx_element] = 0xFFFFFFFF; // go invalid

        alloc_count_ --;
        free_list_chunks_[alloc_count_ / elements_in_chunk_][alloc_count_ % elements_in_chunk_] = idx;

        if (THREAD_SAFE) {
            spin_lock_.unlock();
        }
    }

    FORCE_INLINE uint32_t GetRidCount() const {
        return alloc_count_;
    }

    void SetDescription(const char *description) {
        description_ = description;
    }

    explicit RID_Alloc(uint32_t target_chunk_byte_size = 65536) {
        elements_in_chunk_ = sizeof(T) > target_chunk_byte_size ? 1 : (target_chunk_byte_size / sizeof(T));
    }

    ~RID_Alloc() override {
        if (alloc_count_) {
            LOG_ERROR("ERROR: {} RID allocations of type '{}' were leaked at exit.",
                      alloc_count_, description_ ? description_ : typeid(T).name());

            for (size_t i = 0; i < max_alloc_; i++) {
                uint64_t validator = validator_chunks_[i / elements_in_chunk_][i % elements_in_chunk_];
                if (validator & 0x80000000) {
                    continue; //uninitialized
                }
                if (validator != 0xFFFFFFFF) {
                    chunks_[i / elements_in_chunk_][i % elements_in_chunk_].~T();
                }
            }
        }

        uint32_t chunk_count = max_alloc_ / elements_in_chunk_;
        for (uint32_t i = 0; i < chunk_count; i++) {
            memfree(chunks_[i]);
            free(validator_chunks_[i]);
            free(free_list_chunks_[i]);
        }

        if (chunks_) {
            free(chunks_);
            free(free_list_chunks_);
            free(validator_chunks_);
        }
    }

private:
    T **chunks_ = nullptr;
    uint32_t **free_list_chunks_ = nullptr;
    uint32_t **validator_chunks_ = nullptr;

    uint32_t elements_in_chunk_;
    uint32_t max_alloc_ = 0;
    uint32_t alloc_count_ = 0;

    const char *description_ = nullptr;

    mutable SpinLock spin_lock_;
};

template <class T, bool THREAD_SAFE = false>
class RID_PtrOwner {
    RID_Alloc<T *, THREAD_SAFE> alloc;

public:
    FORCE_INLINE RID MakeRID(T *ptr) {
        return alloc.MakeRID(ptr);
    }

    FORCE_INLINE RID AllocateRID() {
        return alloc.AllocateRID();
    }

    FORCE_INLINE void InitializeRID(RID rid, T *ptr) {
        alloc.InitializeRID(rid, ptr);
    }

    FORCE_INLINE T *GetOrNull(const RID &rid) {
        T **ptr = alloc.GetOrNull(rid);
        if (!ptr) [[unlikely]] {
            return nullptr;
        }
        return *ptr;
    }

    FORCE_INLINE void Replace(const RID &rid, T *new_ptr) {
        T **ptr = alloc.GetOrNull(rid);
        ERR_FAIL_COND(!ptr);
        *ptr = new_ptr;
    }

    FORCE_INLINE bool Owns(const RID &rid) {
        return alloc.Owns(rid);
    }

    FORCE_INLINE void Free(const RID &rid) {
        return alloc.Free(rid);
    }

    [[nodiscard]] FORCE_INLINE uint32_t GetRidCount() const {
        return alloc.GetRidCount();
    }

    void SetDescription(const char *description) {
        alloc.SetDescription(description);
    }

    explicit RID_PtrOwner(uint32_t target_chunk_byte_size = 65535) :
        alloc(target_chunk_byte_size) {}
};

template <class T, bool THREAD_SAFE = false>
class RID_Owner {
    RID_Alloc<T, THREAD_SAFE> alloc;

public:
    FORCE_INLINE RID MakeRID() {
        return alloc.MakeRID();
    }

    FORCE_INLINE RID MakeRID(const T &ptr) {
        return alloc.MakeRID(ptr);
    }

    FORCE_INLINE RID AllocateRID() {
        return alloc.AllocateRID();
    }

    FORCE_INLINE void InitializeRID(RID rid) {
        alloc.InitializeRID(rid);
    }

    FORCE_INLINE void InitializeRID(RID rid, const T &ptr) {
        alloc.InitializeRID(rid, ptr);
    }

    FORCE_INLINE T *GetOrNull(const RID &rid) {
        return alloc.GetOrNull(rid);
    }

    [[nodiscard]] FORCE_INLINE bool Owns(const RID &rid) const {
        return alloc.Owns(rid);
    }

    FORCE_INLINE void Free(const RID &rid) {
        alloc.Free(rid);
    }

    [[nodiscard]] FORCE_INLINE uint32_t GetRidCount() const {
        return alloc.GetRidCount();
    }

    void SetDescription(const char *description) {
        alloc.SetDescription(description);
    }

    explicit RID_Owner(uint32_t target_chunk_byte_size = 65536) :
            alloc(target_chunk_byte_size) {}
};

} // End of namespace gobot