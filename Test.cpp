#include <iostream>
#include <arm_neon.h>
#include <cstdlib>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#define TIME_START start = std::chrono::steady_clock::now();
#define TIME_END(NAME)                                                                     \
    end = std::chrono::steady_clock::now();                                                \
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
    cout << (NAME) << ", duration = " << duration << "ms" << endl;
static int inputSize = 500;
static int kernelSize = 16;
static int kernelNum = 2;
static int outSize = inputSize - (kernelSize - 1);
static void show(float **toShow, int h, int l);
static void show_16(__fp16 **toShow, int h, int l);
__fp16 dotproduct_simd_16(const __fp16 *p1, const __fp16 *p2, size_t n)
{
    if (n % 8 != 0)
    {
        return __fp16(0.0f);
    }

    __fp16 sum[8] = {0.0f};
    float16x8_t a, b;
    float16x8_t c = vdupq_n_f16(0.0f);

    for (size_t i = 0; i < n; i += 8)
    {
        a = vld1q_f16(p1 + i);
        b = vld1q_f16(p2 + i);
        c = vaddq_f16(c, vmulq_f16(a, b));
    }

    vst1q_f16(sum, c);

    return (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]);
}
float dotproduct_simd_32(const float *p1, const float *p2, size_t n)
{
    if (n % 4 != 0)
    {
        std::cerr << "The size n must be a multiple of 4." << std::endl;
        return 0.0f;
    }
    float *sum = new float[4];
    float32x4_t a, b;
    float32x4_t c = vdupq_n_f32(0);

    for (size_t i = 0; i < n; i += 4)
    {
        a = vld1q_f32(p1 + i);
        b = vld1q_f32(p2 + i);
        c = vaddq_f32(c, vmulq_f32(a, b));
    }
    vst1q_f32(sum, c);
    return (sum[0] + sum[1] + sum[2] + sum[3]);
}
static void convDir(float **input, float ***kernel, float ***output)
{
    for (int i = 0; i < kernelNum; i++)
    {
        // each kernel
        int hs = 0;
        int ls = 0;
        // 确定起始点 也是输出的index
        for (int h = hs; h + kernelSize <= inputSize; h++)
        {
            for (int l = ls; l + kernelSize <= inputSize; l++)
            {
                // conv
                float out = 0;
                for (int x = 0; x < kernelSize; x++)
                {
                    for (int y = 0; y < kernelSize; y++)
                    {
                        out += input[h + x][l + y] * kernel[i][x][y];
                    }
                }
                output[i][h][l] = out;
            }
        }
    }
}

static void convIm2rowWithoutSimd(float **input, float ***kernel, float **output)
{
    // 先构建出矩阵
    float **inputC = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        inputC[i] = new float[kernelSize * kernelSize]; // 记得先分配内存！
        // 矩阵的每一行
        int h = i / outSize;
        int l = i % outSize;
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            inputC[i][j] = input[h + (j / kernelSize)][l + (j % kernelSize)];
        }
    }
    // show(inputC, outSize * outSize, kernelSize * kernelSize);
    float **kernelC = new float *[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        kernelC[i] = new float[kernelSize * kernelSize];
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            kernelC[i][j] = kernel[i][j / kernelSize][j % kernelSize];
        }
    }
    // show(kernelC, kernelNum, kernelSize * kernelSize);
    // 矩阵乘法
    for (int h = 0; h < outSize * outSize; h++)
    { // inputC的行
        for (int l = 0; l < kernelNum; l++)
        { // kernelC的列
            float ans = 0;
            for (int k = 0; k < kernelSize * kernelSize; k++)
            {
                ans += inputC[h][k] * kernelC[l][k];
            }
            output[h][l] = ans;
        }
    }
    // show(output, outSize * outSize, kernelNum);
}
static void convIm2rowWithSimd(float **input, float ***kernel, float **output)
{
    // 先构建出矩阵
    float **inputC = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        inputC[i] = new float[kernelSize * kernelSize]; // 记得先分配内存！
        // 矩阵的每一行
        int h = i / outSize;
        int l = i % outSize;
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            inputC[i][j] = input[h + (j / kernelSize)][l + (j % kernelSize)];
        }
    }
    // show(inputC, outSize * outSize, kernelSize * kernelSize);
    float **kernelC = new float *[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        kernelC[i] = new float[kernelSize * kernelSize];
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            kernelC[i][j] = kernel[i][j / kernelSize][j % kernelSize];
        }
    }
    // show(kernelC, kernelNum, kernelSize * kernelSize);
    // 矩阵乘法
    for (int h = 0; h < outSize * outSize; h++)
    { // inputC的行
        for (int l = 0; l < kernelNum; l++)
        { // kernelC的列
            float ans = 0;
            ans = dotproduct_simd_32(inputC[h], kernelC[l], kernelSize * kernelSize);
            output[h][l] = ans;
        }
    }
    // show(output, outSize * outSize, kernelNum);
}

static void convIm2rowFastWithoutSimd(float **input, float ***kernel, float **output)
{
    // 构建矩阵
    // i /(k*(i-k+1))
    float **inputC = new float *[outSize];
    int length = kernelSize * kernelSize + ((inputSize - kernelSize) * kernelSize); // 表示每一行的长度
    for (int i = 0; i < outSize; i++)
    {
        inputC[i] = new float[length];
        int h = 0;
        int l = i;
        int exterH = kernelSize; // 代表多余部分位于input的哪一行
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            inputC[i][j] = input[h + (j / kernelSize)][l + (j % kernelSize)];
        }
        for (int j = kernelSize * kernelSize; j < length; j += kernelSize)
        {
            // 每kernel个 对应到一行 开始的位置看i
            for (int k = 0; k < kernelSize; k++)
            {
                inputC[i][j + k] = input[exterH][i + k];
            }
            exterH++;
        }
    }
    float **kernelC = new float *[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        kernelC[i] = new float[kernelSize * kernelSize];
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            kernelC[i][j] = kernel[i][j / kernelSize][j % kernelSize];
        }
    }
    // show(kernelC, kernelNum, kernelSize * kernelSize);

    // 矩阵乘法
    // inputC的行
    for (int i = 0; i < outSize; i++)
    {
        for (int h = 0; h < outSize; h++)
        {
            // kernelC的列
            for (int l = 0; l < kernelNum; l++)
            {
                float ans = 0;
                for (int k = 0; k < kernelSize * kernelSize; k++)
                {
                    ans += inputC[h][k + i * kernelSize] * kernelC[l][k];
                }
                output[h + i * outSize][l] = ans;
            }
        }
    }
    // show(output, outSize * outSize, kernelNum);
}

static void convIm2rowFastWithSimd(float **input, float ***kernel, float **output)
{
    // 构建矩阵
    // i /(k*(i-k+1))
    float **inputC = new float *[outSize];
    int length = kernelSize * kernelSize + ((inputSize - kernelSize) * kernelSize); // 表示每一行的长度
    for (int i = 0; i < outSize; i++)
    {
        inputC[i] = new float[length];
        int h = 0;
        int l = i;
        int exterH = kernelSize; // 代表多余部分位于input的哪一行
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            inputC[i][j] = input[h + (j / kernelSize)][l + (j % kernelSize)];
        }
        for (int j = kernelSize * kernelSize; j < length; j += kernelSize)
        {
            // 每kernel个 对应到一行 开始的位置看i
            for (int k = 0; k < kernelSize; k++)
            {
                inputC[i][j + k] = input[exterH][i + k];
            }
            exterH++;
        }
    }
    float **kernelC = new float *[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        kernelC[i] = new float[kernelSize * kernelSize];
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            kernelC[i][j] = kernel[i][j / kernelSize][j % kernelSize];
        }
    }
    // show(kernelC, kernelNum, kernelSize * kernelSize);

    // 矩阵乘法
    // inputC的行
    for (int i = 0; i < outSize; i++)
    {
        for (int h = 0; h < outSize; h++)
        {
            // kernelC的列
            for (int l = 0; l < kernelNum; l++)
            {
                float ans = 0;
                // for (int k = 0; k < kernelSize * kernelSize; k++)
                // {
                //     ans += inputC[h][k + i * kernelSize] * kernelC[l][k];
                // }
                // TODO
                ans = dotproduct_simd_32(inputC[h] + i * kernelSize, kernelC[l], kernelSize * kernelSize);
                output[h + i * outSize][l] = ans;
            }
        }
    }
    // show(output, outSize * outSize, kernelNum);
}

static void convIm2rowFastWithSimd_16(__fp16 **input, __fp16 ***kernel, __fp16 **output)
{
    // 构建矩阵
    // i /(k*(i-k+1))
    __fp16 **inputC = new __fp16 *[outSize];
    int length = kernelSize * kernelSize + ((inputSize - kernelSize) * kernelSize); // 表示每一行的长度
    for (int i = 0; i < outSize; i++)
    {
        inputC[i] = new __fp16[length];
        int h = 0;
        int l = i;
        int exterH = kernelSize; // 代表多余部分位于input的哪一行
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            inputC[i][j] = input[h + (j / kernelSize)][l + (j % kernelSize)];
        }
        for (int j = kernelSize * kernelSize; j < length; j += kernelSize)
        {
            // 每kernel个 对应到一行 开始的位置看i
            for (int k = 0; k < kernelSize; k++)
            {
                inputC[i][j + k] = input[exterH][i + k];
            }
            exterH++;
        }
    }
    __fp16 **kernelC = new __fp16 *[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        kernelC[i] = new __fp16[kernelSize * kernelSize];
        for (int j = 0; j < kernelSize * kernelSize; j++)
        {
            kernelC[i][j] = kernel[i][j / kernelSize][j % kernelSize];
        }
    }
    // show(kernelC, kernelNum, kernelSize * kernelSize);

    // 矩阵乘法
    // inputC的行
    for (int i = 0; i < outSize; i++)
    {
        for (int h = 0; h < outSize; h++)
        {
            // kernelC的列
            for (int l = 0; l < kernelNum; l++)
            {
                __fp16 ans = 0;
                // for (int k = 0; k < kernelSize * kernelSize; k++)
                // {
                //     ans += inputC[h][k + i * kernelSize] * kernelC[l][k];
                // }
                // TODO
                ans = dotproduct_simd_16(inputC[h] + i * kernelSize, kernelC[l], kernelSize * kernelSize);
                output[h + i * outSize][l] = ans;
            }
        }
    }
    // show_16(output, outSize * outSize, kernelNum);
}

// 工具函数
static void fill(float **toFill, int size, float value)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            toFill[i][j] = value;
        }
    }
}

static void fill_16(__fp16 **toFill, int size, __fp16 value)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            toFill[i][j] = value;
        }
    }
}

static void show(float **toShow, int h, int l)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < l; j++)
        {
            cout << toShow[i][j] << " ";
        }
        cout << endl;
    }
    cout << "-----------------" << endl;
}

static void show_16(__fp16 **toShow, int h, int l)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < l; j++)
        {
            cout << toShow[i][j] << " ";
        }
        cout << endl;
    }
    cout << "-----------------" << endl;
}

static float **creatArray2(int size)
{
    float **array2 = new float *[size];
    for (int i = 0; i < size; i++)
    {
        array2[i] = new float[size];
    }
    return array2;
}

static __fp16 **creatArray2_16(int size)
{
    __fp16 **array2 = new __fp16 *[size];
    for (int i = 0; i < size; i++)
    {
        array2[i] = new __fp16[size];
    }
    return array2;
}
// static void deleteArray2(float **input, int inputSize)
// {
//     for (int i = 0; i < inputSize; i++)
//     {
//         delete[] input[i];
//     }
//     delete[] input;
// }
int main()
{
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;
    // 预热
    for (long i = 0; i < 100000000L; i++)
    {
        double x = (((i + 0.1) * 1.25) - 9.9) / 3.33;
    }

    // 准备16的数据
    __fp16 **input_16 = creatArray2_16(inputSize);
    __fp16 **kernel1_16 = creatArray2_16(kernelSize);
    __fp16 **kernel2_16 = creatArray2_16(kernelSize);
    fill_16(input_16, inputSize, 6.0);
    fill_16(kernel1_16, kernelSize, 10.0);
    fill_16(kernel2_16, kernelSize, 15.0);
    __fp16 ***kernel_16 = new __fp16 **[2];
    kernel_16[0] = kernel1_16;
    kernel_16[1] = kernel2_16;

    // 准备数据
    float **input = creatArray2(inputSize);
    float **kernel1 = creatArray2(kernelSize);
    float **kernel2 = creatArray2(kernelSize);

    fill(input, inputSize, 6.0);
    fill(kernel1, kernelSize, 10.0);
    fill(kernel2, kernelSize, 15.0);
    float ***kernel = new float **[2];
    kernel[0] = kernel1;
    kernel[1] = kernel2;

    TIME_START
    // normal conv
    float ***outputDir = new float **[kernelNum];
    for (int i = 0; i < kernelNum; i++)
    {
        outputDir[i] = creatArray2(outSize);
    }
    convDir(input, kernel, outputDir);
    // show(outputDir[0], outSize, outSize);
    // show(outputDir[1], outSize, outSize);
    TIME_END("normal conv")

    TIME_START
    // conv Im2rowWithoutSimd
    float **outputIm2rowWithoutSimd = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        outputIm2rowWithoutSimd[i] = new float[kernelNum];
    }
    convIm2rowWithoutSimd(input, kernel, outputIm2rowWithoutSimd);
    TIME_END("Im2rowWithoutSimd")

    TIME_START
    // conv Im2rowWithSimd
    float **outputIm2rowWithSimd = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        outputIm2rowWithSimd[i] = new float[kernelNum];
    }
    convIm2rowWithSimd(input, kernel, outputIm2rowWithSimd);
    TIME_END("Im2rowWithSimd")

    TIME_START
    // conv Im2rowFastWithOutSimd
    float **outputIm2rowFastWithOutSimd = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        outputIm2rowWithoutSimd[i] = new float[kernelNum];
    }
    convIm2rowFastWithoutSimd(input, kernel, outputIm2rowWithoutSimd);
    TIME_END("Im2rowFastWithOutSimd")

    TIME_START
    // conv Im2rowFastWithSimd
    float **outputIm2rowFastWithSimd = new float *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        outputIm2rowFastWithSimd[i] = new float[kernelNum];
    }
    convIm2rowFastWithSimd(input, kernel, outputIm2rowFastWithSimd);
    TIME_END("Im2rowFastWithSimd");

    TIME_START
    // conv Im2rowFastWithSimd_16
    __fp16 **outputIm2rowFastWithSimd_16 = new __fp16 *[outSize * outSize];
    for (int i = 0; i < outSize * outSize; i++)
    {
        outputIm2rowFastWithSimd_16[i] = new __fp16[kernelNum];
    }
    convIm2rowFastWithSimd_16(input_16, kernel_16, outputIm2rowFastWithSimd_16);
    TIME_END("Im2rowFastWithSimd_16");

    return 0;
}
