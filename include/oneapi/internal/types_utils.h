/* file: types_utils.h */
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

#ifndef __DAAL_ONEAPI_INTERNAL_TYPES_UTILS_H__
#define __DAAL_ONEAPI_INTERNAL_TYPES_UTILS_H__

#include "oneapi/internal/types.h"
#ifdef DAAL_SYCL_INTERFACE
    #include <CL/sycl.hpp>
#endif

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

template <typename...>
struct Typelist
{};

typedef Typelist<daal::oneapi::internal::int8_t, daal::oneapi::internal::int16_t, daal::oneapi::internal::int32_t, daal::oneapi::internal::int64_t,
                 daal::oneapi::internal::uint8_t, daal::oneapi::internal::uint16_t, daal::oneapi::internal::uint32_t,
                 daal::oneapi::internal::uint64_t, daal::oneapi::internal::float32_t, daal::oneapi::internal::float64_t>
    PrimitiveTypes;

typedef Typelist<daal::oneapi::internal::float32_t, daal::oneapi::internal::float64_t> FloatTypes;

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__TYPEDISPATCHER"></a>
 *  \brief Makes runtime dispatching of types
 */
class TypeDispatcher
{
public:
    template <typename Operation>
    static auto dispatch(TypeId type, Operation && op) -> void
    {
        return dispatchInternal(type, op, PrimitiveTypes());
    }

    template <typename Operation>
    static auto floatDispatch(TypeId type, Operation && op) -> void
    {
        return dispatchInternal(type, op, FloatTypes());
    }

#ifdef DAAL_SYCL_INTERFACE
    template <typename Operation>
    static auto dispatch(TypeId type, Operation && op) -> cl::sycl::event
    {
        return dispatchInternal(type, op, PrimitiveTypes());
    }

    template <typename Operation>
    static auto floatDispatch(TypeId type, Operation && op) -> cl::sycl::event
    {
        return dispatchInternal(type, op, FloatTypes());
    }
#endif

private:
    template <typename Operation, typename Head, typename... Rest>
    static auto dispatchInternal(TypeId type, Operation && op, Typelist<Head, Rest...>) -> void
    {
        if (type == TypeIds::id<Head>())
        {
            return op(Typelist<Head>());
        }
        else
        {
            return dispatchInternal(type, op, Typelist<Rest...>());
        }
    }

    template <typename Operation>
    static auto dispatchInternal(TypeId type, Operation && op, Typelist<>) -> void
    {
        DAAL_ASSERT(!"Unknown type");
        return;
    }

#ifdef DAAL_SYCL_INTERFACE
    template <typename Operation, typename Head, typename... Rest>
    static auto dispatchInternal(TypeId type, Operation && op, Typelist<Head, Rest...>) -> cl::sycl::event
    {
        if (type == TypeIds::id<Head>())
        {
            return op(Typelist<Head>());
        }
        else
        {
            return dispatchInternal(type, op, Typelist<Rest...>());
        }
    }

    template <typename Operation>
    static auto dispatchInternal(TypeId type, Operation && op, Typelist<>) -> cl::sycl::event
    {
        DAAL_ASSERT(!"Unknown type");
        return cl::sycl::event {};
    }
#endif
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__TYPETOSTRINGCONVERTER"></a>
 *  \brief Converts type to string representation
 */
struct TypeToStringConverter
{
    services::String result;

    template <typename T>
    void operator()(Typelist<T>)
    {
        result = daal::oneapi::internal::getKeyFPType<T>();
    }
};

services::String getKeyFPType(TypeId typeId);

/** @} */

} // namespace interface1

using interface1::Typelist;
using interface1::TypeDispatcher;
using interface1::getKeyFPType;

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif