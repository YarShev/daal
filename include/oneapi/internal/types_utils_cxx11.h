/* file: types_utils_cxx11.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __DAAL_ONEAPI_INTERNAL_TYPES_UTILS_CXX11_H__
#define __DAAL_ONEAPI_INTERNAL_TYPES_UTILS_CXX11_H__

#include "oneapi/internal/types_utils.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERALLOCATOR"></a>
 *  \brief Allocator for UniversalBuffer
 */
class BufferAllocator
{
private:
    struct Allocate
    {
        UniversalBuffer buffer;
        size_t bufferSize;

        explicit Allocate(size_t size) : bufferSize(size) {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            buffer = services::Buffer<T>(cl::sycl::buffer<T, 1>(bufferSize));
            return;
        }
    };

public:
    static UniversalBuffer allocate(TypeId type, size_t bufferSize)
    {
        Allocate allocateOp(bufferSize);
        TypeDispatcher<SyclEventNoExist>::dispatch(type, allocateOp);
        return allocateOp.buffer;
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERCOPIER"></a>
 *  \brief Copier for two UniversalBuffers
 */
class BufferCopier
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        UniversalBuffer & srcUnivers;
        size_t srcOffset;
        size_t count;
        bool isSync;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, UniversalBuffer & src, size_t srcOffset, size_t count,
                         bool isSync = true)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcUnivers(src), srcOffset(srcOffset), count(count), isSync(isSync)
        {}

        template <typename T>
        cl::sycl::event operator()(Typelist<T>)
        {
            auto src              = srcUnivers.get<T>().toSycl();
            auto dst              = dstUnivers.get<T>().toSycl();
            cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                auto src_acc = src.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(srcOffset));
                auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                cgh.copy(src_acc, dst_acc);
            });
            if (isSync)
            {
                event.wait();
            }
            else
            {
                return event;
            }
        }
    };

public:
    static cl::sycl::event copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, UniversalBuffer & src, size_t srcOffset,
                                size_t count, bool isSync = true)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count, isSync);
        return TypeDispatcher<SyclEventExist>::dispatch(dest.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__ARRAYCOPIER"></a>
 *  \brief Copier from array to UniversalBuffers
 */
class ArrayCopier
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        size_t dstOffset;
        void * srcArray;
        size_t srcOffset;
        size_t count;
        bool isSync;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dst, size_t desOffset, void * src, size_t srcOffset, size_t count,
                         bool isSync = true)
            : queue(queue), dstUnivers(dst), dstOffset(desOffset), srcArray(src), srcOffset(srcOffset), count(count), isSync(isSync)
        {}

        template <typename T>
        cl::sycl::event operator()(Typelist<T>)
        {
            auto src              = (T *)srcArray;
            auto dst              = dstUnivers.get<T>().toSycl();
            cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                auto dst_acc = dst.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>(count), cl::sycl::id<1>(dstOffset));
                cgh.copy(src, dst_acc);
            });
            if (isSync)
            {
                event.wait();
            }
            else
            {
                return event;
            }
        }
    };

public:
    static cl::sycl::event copy(cl::sycl::queue & queue, UniversalBuffer & dest, size_t dstOffset, void * src, size_t srcOffset, size_t count,
                                bool isSync = true)
    {
        Execute op(queue, dest, dstOffset, src, srcOffset, count, isSync);
        return TypeDispatcher<SyclEventExist>::dispatch(dest.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__BUFFERFILLER"></a>
 *  \brief Fills UniversalBuffers with single value
 */
class BufferFiller
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        UniversalBuffer & dstUnivers;
        double value;
        bool isSync;

        explicit Execute(cl::sycl::queue & queue, UniversalBuffer & dest, double value, bool isSync = true)
            : queue(queue), dstUnivers(dest), value(value), isSync(isSync)
        {}

        template <typename T>
        cl::sycl::event operator()(Typelist<T>)
        {
            auto dst              = dstUnivers.get<T>().toSycl();
            cl::sycl::event event = queue.submit([&](cl::sycl::handler & cgh) {
                auto acc = dst.template get_access<cl::sycl::access::mode::write>(cgh);
                cgh.fill(acc, static_cast<T>(value));
            });
            if (isSync)
            {
                event.wait();
            }
            else
            {
                return event;
            }
        }
    };

public:
    static cl::sycl::event fill(cl::sycl::queue & queue, UniversalBuffer & dest, double value, bool isSync = true)
    {
        Execute op(queue, dest, value, isSync);
        return TypeDispatcher<SyclEventExist>::dispatch(dest.type(), op);
    }
};

/** @} */
} // namespace interface1

using interface1::BufferAllocator;
using interface1::BufferCopier;
using interface1::BufferFiller;

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
