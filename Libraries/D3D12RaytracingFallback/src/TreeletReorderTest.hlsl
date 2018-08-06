//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

// Using the Karras/Aila paper on treelet reoordering:
// "Fast Parallel Construction of High-Quality Bounding Volume
// Hierarchies"

#define HLSL
#include "TreeletReorderBindings.h"
#include "RayTracingHelper.hlsli"

// Must be at least FullTreeletSize
#define NumThreadsInGroup 32

groupshared uint nodeIndex;
groupshared float optimalCost[NumTreeletSplitPermutations];
groupshared uint optimalPartition[NumTreeletSplitPermutations];
groupshared uint treeletToReorder[FullTreeletSize];
groupshared uint internalNodes[NumInternalTreeletNodes];
groupshared bool finished;

void FindOptimalPartitions(in uint threadId)
{
    uint numBitmasksPerThread;
    uint extraBitmasks;
    uint bitmasksStart;
    uint bitmasksEnd;

    uint debugCursor = 0;

    // Dynamic programming from 'treelet/subset' of size 2 up to FullTreeletSize, calculate and store optimal (lowest) cost and its partition bitmask
    for (uint subsetSize = 2; subsetSize <= FullTreeletSize; subsetSize++)
    {
        // eg. In 'treelet/subset' of size 2, there are (7 Choose 2) distinct 'treelets' in the original treelet of 7 leaves, ie. 0000011, 0000101, ..., 1100000
        uint numTreeletBitmasks = FullTreeletSizeChoose[subsetSize];
        if (threadId < numTreeletBitmasks)
        {
            numBitmasksPerThread = max(numTreeletBitmasks / NumThreadsInGroup, 1);
            extraBitmasks = numTreeletBitmasks > NumThreadsInGroup ? numTreeletBitmasks % NumThreadsInGroup : 0;

            bitmasksStart = numBitmasksPerThread * threadId + min(threadId, extraBitmasks);
            bitmasksEnd = bitmasksStart + numBitmasksPerThread + (threadId < extraBitmasks ? 1 : 0);

            // For each subset with [subsetSize] bits set
            for (uint i = bitmasksStart; i < bitmasksEnd; i++)
            {
                uint treeletBitmask = GetBitPermutation(subsetSize, i);

                float lowestCost = FLT_MAX;
                uint bestPartition = 0;

                uint delta = (treeletBitmask - 1) & treeletBitmask;
                uint partitionBitmask = (-delta) & treeletBitmask;

                do
                {
                    partitionBitmask = (partitionBitmask - delta) & treeletBitmask;

                    if (threadId == 0)
                    {
                        ASDebugLog[debugCursor] = partitionBitmask;
                    }
                    if (++debugCursor >= AccelerationStructureDebugLogLength) return;

                } while (partitionBitmask != 0);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(NumThreadsInGroup, 1, 1)]
void main(uint3 Gid : SV_GroupID, uint3 GTid : SV_GroupThreadId)
{
    if (GTid.x == 0)
    {
        nodeIndex = BaseTreeletsIndexBuffer[Gid.x];
        finished = false;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint NumberOfAABBs = GetNumInternalNodes(Constants.NumberOfElements) + Constants.NumberOfElements;

    if (nodeIndex >= NumberOfAABBs)
    {
        return;
    }

    FindOptimalPartitions(GTid.x);
}