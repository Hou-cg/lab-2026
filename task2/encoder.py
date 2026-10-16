# encoder.py
# 姓名：侯程罡    学号：202600192
# 程序简介：把 PPM（P6 二进制）图片编码成 OKTI 文本格式。
#           本版本使用 p（完整 RGB）、d（差值）、r/R（重复上一个像素）、
#           i/I（颜色表索引）六种标记（i 与 I、r 与 R 各是一种写法的长短两版）。
#           输出文件名固定为 output.okti。

import os  # 用于拼接"与本脚本同目录"的路径，避免受运行时所在目录影响
import sys  # 用于读取命令行参数

# ===== 常量（永不改变的值，放成常量可以避免"魔术数字"被扣分）=====
MAGIC_NUMBER = b"P6"  # PPM 二进制格式的标识，文件必须以这两个字符开头
MAX_COLOR_VALUE = 255  # PPM 里每个颜色分量的最大值（超出这个值就不是 8 位色）
COLOR_COMPONENT_COUNT = 3  # 每个像素由红、绿、蓝 3 个分量组成
MAX_SHORT_RUN = 15  # 小写 r 后面是 1 位十六进制数字，最多表示重复 15 次
MAX_LONG_RUN = 255  # 大写 R 后面是 2 位十六进制数字，最多表示重复 255 次
SHORT_RUN_CHAR = "r"  # 短重复的标记字符（小写）
LONG_RUN_CHAR = "R"  # 长重复的标记字符（大写）
FULL_PIXEL_CHAR = "p"  # 完整 RGB 像素的标记字符
FULL_PIXEL_LENGTH = 7  # "p" 加 6 位十六进制数字，共 7 个字符
DIFFERENCE_CHAR = "d"  # 与上一个像素的差值编码的标记字符（小写）
DIFFERENCE_BIAS = 8  # 差值编码的偏移量：真实差值是 -8~+7，写出来要统一加 8（即 0~15）
MIN_DIFFERENCE = -DIFFERENCE_BIAS  # 差值允许的最小值 -8（写成十六进制 0）
MAX_DIFFERENCE = 15 - DIFFERENCE_BIAS  # 差值允许的最大值 +7（写成十六进制 f）
SHORT_INDEX_CHAR = "i"  # 颜色表索引的短写法（小写），索引范围 0~15
LONG_INDEX_CHAR = "I"  # 颜色表索引的长写法（大写），索引范围 0~255
MAX_SHORT_INDEX = 15  # 小写 i 后面是 1 位十六进制数字，最大索引 15
MAX_COLOR_TABLE_SIZE = 256  # 颜色表最多保存 256 种颜色
INITIAL_COLOR = (0, 0, 0)  # 颜色表的初始内容：纯黑（按 OKTI 规定）
OUTPUT_FILE_NAME = "output.okti"  # 作业固定要求写死的输出文件名（不要在代码里从用户处读取）


def skip_whitespace_and_comments(data, index):
    """跳过 PPM 头部的空白字符和以 # 开头的注释行。

    参数 data：整个文件的字节串。
    参数 index：从哪个位置开始往下看。
    返回：第一个"不是空白也不是注释"的位置。
    """
    position = index  # 用一个新变量记录当前位置，避免改动传进来的参数
    scanning = True  # 循环开关：找到非空白内容时置为 False 结束循环
    while scanning and position < len(data):  # 只要还没看完整份数据就继续
        current = data[position]  # 取出当前位置的那一个字节
        if current in b" \t\r\n":  # 如果是空格、制表符或换行，就当作空白
            position = position + 1  # 跳过这一个空白字符
        elif current == ord("#"):  # 如果是井号，说明这一行是注释
            while position < len(data) and data[position] != ord("\n"):  # 一直读到行尾
                position = position + 1  # 逐个字符往后移
        else:  # 既不是空白也不是注释，说明找到有效内容了
            scanning = False  # 关掉循环开关，准备返回
    return position  # 返回有效内容的起始位置


def read_ppm_token(data, index):
    """从 PPM 头部读出一个"以空白分隔的字段"（例如宽度、高度、最大值）。

    参数 data：整个文件的字节串。
    参数 index：开始读的位置。
    返回：一个二元组（字段的字节串, 字段结束后的位置）。
    """
    index = skip_whitespace_and_comments(data, index)  # 先跳过空白和注释
    start = index  # 记下字段的起点
    while index < len(data) and data[index] not in b" \t\r\n":  # 一直读到空白为止
        index = index + 1  # 每读一个字符就往后移一位
    return data[start:index], index  # 返回字段内容，以及停下来的位置


def read_ppm(path):
    """读取一个 PPM（P6 二进制）图片文件。

    参数 path：要读取的文件路径。
    返回：一个字典，含 width（宽）、height（高）和 pixels（像素列表，
          每个像素是 (红, 绿, 蓝) 三个 0~255 的整数）。
    """
    with open(path, "rb") as ppm_file:  # 以二进制方式打开，因为像素数据不是文字
        data = ppm_file.read()  # 一次把整个文件读进内存，方便按位置解析

    magic, index = read_ppm_token(data, 0)  # 第 1 个字段是格式标识
    if magic != MAGIC_NUMBER:  # 不是 "P6" 说明格式不对
        raise ValueError("这个文件不是 P6 类型的 PPM 文件：" + path)

    width_text, index = read_ppm_token(data, index)  # 第 2 个字段是宽度
    height_text, index = read_ppm_token(data, index)  # 第 3 个字段是高度
    max_value_text, index = read_ppm_token(data, index)  # 第 4 个字段是颜色最大值

    width = int(width_text)  # 把宽度从字节串转成整数
    height = int(height_text)  # 把高度从字节串转成整数
    max_value = int(max_value_text)  # 把颜色最大值转成整数

    if width <= 0 or height <= 0:  # 宽或高不是正整数就说明图片非法
        raise ValueError("图片尺寸非法：宽 " + str(width) + "，高 " + str(height))
    if max_value != MAX_COLOR_VALUE:  # 只有最大值 255 才是一个分量占 1 字节
        raise ValueError("暂不支持颜色最大值 " + str(max_value) + " 的 PPM 文件")

    # 头部字段之间只要求有一个空白字符，多出来的空白不属于像素数据，必须跳过
    index = skip_whitespace_and_comments(data, index + 1)

    needed_bytes = width * height * COLOR_COMPONENT_COUNT  # 总共需要多少个字节的像素数据
    pixel_data = data[index:]  # 从当前位置到文件末尾，就是像素数据
    if len(pixel_data) < needed_bytes:  # 数据不够说明文件被截断了
        raise ValueError("像素数据不足：需要 " + str(needed_bytes) + " 字节，实际只有 " + str(len(pixel_data)) + " 字节")

    pixels = []  # 用来存放解析出来的像素，顺序是从左到右、从上到下
    offset = 0  # 指向当前要读的像素数据的起始位置
    while offset < needed_bytes:  # 逐个像素读取，直到读满宽 × 高 个像素
        red = pixel_data[offset]  # 第 1 个字节是红色分量
        green = pixel_data[offset + 1]  # 第 2 个字节是绿色分量
        blue = pixel_data[offset + 2]  # 第 3 个字节是蓝色分量
        pixels.append((red, green, blue))  # 把这三个分量作为一个像素存起来
        offset = offset + COLOR_COMPONENT_COUNT  # 往后移 3 个字节，准备读下一个像素

    return {"width": width, "height": height, "pixels": pixels}  # 返回解析结果


def full_pixel_token(red, green, blue):
    """把红绿蓝三个分量拼成"完整 RGB 像素"的编码，例如 pff0000。

    参数 red：红色分量（0~255）。
    参数 green：绿色分量（0~255）。
    参数 blue：蓝色分量（0~255）。
    返回：7 个字符的编码字符串。
    """
    red_text = "%02x" % red  # %02x 表示转成 2 位十六进制，不足两位时前面补 0
    green_text = "%02x" % green  # 绿色分量同理
    blue_text = "%02x" % blue  # 蓝色分量同理
    return FULL_PIXEL_CHAR + red_text + green_text + blue_text  # 用加号把 1 + 6 个字符接起来


def difference_token(red_difference, green_difference, blue_difference):
    """把三个通道的差值拼成"与上一个像素的差值"编码，例如 d08f。

    参数 red_difference：红色分量相对上一个像素的差值（-8 ~ +7）。
    参数 green_difference：绿色分量相对上一个像素的差值（-8 ~ +7）。
    参数 blue_difference：蓝色分量相对上一个像素的差值（-8 ~ +7）。
    返回：4 个字符的编码字符串（d 加 3 位十六进制数字）。
    """
    red_text = "%x" % (red_difference + DIFFERENCE_BIAS)  # 差值加 8 变成 0~15，再写成 1 位十六进制
    green_text = "%x" % (green_difference + DIFFERENCE_BIAS)  # 绿色通道同理
    blue_text = "%x" % (blue_difference + DIFFERENCE_BIAS)  # 蓝色通道同理
    return DIFFERENCE_CHAR + red_text + green_text + blue_text  # 拼成 d + 3 位十六进制


def color_index(color_table, color):
    """在颜色表里查找某个颜色所在的位置。

    参数 color_table：颜色表（列表），最新的颜色在最前面。
    参数 color：要查找的颜色，形如 (红, 绿, 蓝)。
    返回：颜色在表里的下标；如果表里没有这个颜色则返回 -1。
    """
    for position in range(len(color_table)):  # 从表头开始逐个比较
        if color_table[position] == color:  # 找到相同颜色
            return position  # 返回它的下标
    return -1  # 找遍了也没有


def index_token(position):
    """把颜色表下标拼成"复制之前出现过的像素"的编码：小时用 i，大时用 I。

    参数 position：颜色在颜色表里的下标（0~255）。
    返回：2 个或 3 个字符的编码字符串，例如 i0、ib、I11、Ia2。
    """
    if position <= MAX_SHORT_INDEX:  # 0~15 用 1 位十六进制数字，不补前导 0
        return SHORT_INDEX_CHAR + ("%x" % position)  # 例如下标 11 写成 ib
    return LONG_INDEX_CHAR + ("%02x" % position)  # 16~255 必须用 2 位十六进制数字，例如下标 17 写成 I11


def is_difference_allowed(previous, current):
    """判断当前像素能否用"与上一个像素的差值"（d）来表示。

    参数 previous：上一个像素，形如 (红, 绿, 蓝)。
    参数 current：当前像素，形如 (红, 绿, 蓝)。
    返回：三个通道的差值是否都在 -8 ~ +7 之间；是则返回 True。
    """
    red_difference = current[0] - previous[0]  # 红色通道的差值
    green_difference = current[1] - previous[1]  # 绿色通道的差值
    blue_difference = current[2] - previous[2]  # 蓝色通道的差值
    if red_difference < MIN_DIFFERENCE or red_difference > MAX_DIFFERENCE:  # 红色差超出范围
        return False  # 不能用 d
    if green_difference < MIN_DIFFERENCE or green_difference > MAX_DIFFERENCE:  # 绿色差超出范围
        return False  # 不能用 d
    if blue_difference < MIN_DIFFERENCE or blue_difference > MAX_DIFFERENCE:  # 蓝色差超出范围
        return False  # 不能用 d
    return True  # 三个差值都在 -8~+7 之间，可以用 d


def run_token(copies):
    """把"重复上一个像素"的编码拼出来：次数小于 16 用小写 r，否则用大写 R。

    参数 copies：要重复的额外份数（1~255）。
    返回：2 个或 3 个字符的编码字符串，例如 r5 或 R20。
    """
    if copies <= MAX_SHORT_RUN:  # 1~15 次可以用 1 位十六进制数字表示
        return SHORT_RUN_CHAR + ("%x" % copies)  # %x 转成十六进制，不需要补前导 0
    return LONG_RUN_CHAR + ("%02x" % copies)  # 16~255 次必须用 2 位十六进制数字表示


def encode_pixels(pixels):
    """把像素序列编码成 OKTI 的编码片段列表。

    参数 pixels：像素列表，每个像素是 (红, 绿, 蓝)。
    返回：字符串列表，每个元素是一段编码（p、d、r、R、i 或 I 中的一种）。
    """
    tokens = []  # 用来按顺序存放每一段编码
    pixel_index = 0  # 指向当前正在处理的像素
    total_pixels = len(pixels)  # 一共有多少个像素
    previous = INITIAL_COLOR  # "上一个像素"，按 OKTI 规定初始为纯黑
    color_table = [INITIAL_COLOR]  # 颜色表，按 OKTI 规定初始只含纯黑

    while pixel_index < total_pixels:  # 一直处理到最后一个像素
        red, green, blue = pixels[pixel_index]  # 取出当前像素的三个分量
        current = (red, green, blue)  # 当前像素的颜色
        red_difference = red - previous[0]  # 红色通道相对上一个像素的差值
        green_difference = green - previous[1]  # 绿色通道的差值
        blue_difference = blue - previous[2]  # 蓝色通道的差值

        # 第一步：先看后面的像素有没有和当前像素颜色完全相同的，能合并成 run 就优先用 r/R
        copies = 0  # 统计后面还有多少个与当前像素颜色完全相同的像素
        same = True  # 循环开关：发现颜色不同就置为 False
        while same and pixel_index + copies + 1 < total_pixels:  # 只允许在图片范围内往后比
            if pixels[pixel_index + copies + 1] == current:  # 与当前像素比颜色
                copies = copies + 1  # 相同，可重复的份数加一
            else:  # 颜色不同，这一段重复到此为止
                same = False  # 关掉循环开关

        # 第二步：决定这一种颜色用哪种写法描述一次（优先级 r/R 已在上面用掉，这里按 i/I、d、p）
        position = color_index(color_table, current)  # 看这个颜色是否已经在颜色表里
        if position >= 0:  # 在表里：用索引写法最省（i 或 I）
            tokens.append(index_token(position))  # 写出 i/I 编码
        elif is_difference_allowed(previous, current):  # 不在表里，但三个差值都在 -8~+7 之间
            tokens.append(difference_token(red_difference, green_difference, blue_difference))  # 用 d 编码
        else:  # 既不在表里，差值也太大，只能写完整 RGB
            tokens.append(full_pixel_token(red, green, blue))  # 用 p 编码

        # 第三步：如果后面有重复的像素，再用 r/R 表示这些追加的份数
        remaining = copies  # 还有多少个相同像素需要靠 r/R 来表示
        while remaining > 0:  # 超过 255 份时要拆成多段 r/R
            if remaining <= MAX_LONG_RUN:  # 剩下的份数一次就能写完
                tokens.append(run_token(remaining))  # 追加一段 r 或 R（r N 表示再复制 N 个像素）
                remaining = 0  # 没有剩余了
            else:  # 剩下的超过 255 份，一次 run 最多只能表示 255 份
                tokens.append(run_token(MAX_LONG_RUN))  # 先写一段 R 表示 255 个
                remaining = remaining - MAX_LONG_RUN  # 再处理剩下的

        # 第四步：把这个颜色登记进颜色表（表里没有才插到最前面），并更新"上一个像素"
        if position < 0:  # 这是一个表里没有的新颜色
            if len(color_table) >= MAX_COLOR_TABLE_SIZE:  # 再插一项就会超过 256 项
                color_table.pop()  # 先丢掉表里最后一项（最久没出现的颜色）
            color_table.insert(0, current)  # 把新颜色插到表的最前面
        previous = current  # 当前像素成为下一个像素的"上一个像素"

        pixel_index = pixel_index + copies + 1  # 当前像素和这些被复制的像素都已处理完
    return tokens  # 返回编码结果


def save_okti(image, path):
    """把图片数据按 OKTI 格式写入文件（第 1 行 okti，第 2 行 宽 高，之后每个编码占一行）。

    参数 image：read_ppm 返回的图片字典。
    参数 path：要写入的文件路径。
    返回：None（这个函数只负责写文件，没有返回值）。
    """
    with open(path, "w", encoding="ascii", newline="\n") as okti_file:  # newline="\n" 关掉 Windows 的行尾自动转换，保证换行只写一个 0A
        okti_file.write("okti" + "\n")  # 第 1 行固定写 okti，表示这是 OKTI 文件
        okti_file.write(str(image["width"]) + " " + str(image["height"]) + "\n")  # 第 2 行写宽度和高度，中间用空格隔开
        for token in encode_pixels(image["pixels"]):  # 逐个取出编码片段
            okti_file.write(token + "\n")  # 每段编码各占一行


def main():
    """主程序：确定输入输出文件名，把 PPM 读进来，再写成 OKTI 文件。

    参数：无（图片文件名来自命令行参数或用户输入）。
    返回：None。
    """
    if len(sys.argv) == 2:  # 命令行恰好给了 1 个参数
        input_path = sys.argv[1]  # 就用它当要打开的图片文件名
    elif len(sys.argv) == 1:  # 没有给任何命令行参数
        input_path = input("请输入要编码的图片文件名：")  # 用 input 函数问用户要文件名
    else:  # 给了 2 个或更多参数
        print("错误：命令行参数太多，最多只能给一个图片文件名。")
        sys.exit(1)  # 结束程序，并用非零状态码表示是异常结束

    output_path = os.path.join(  # 输出固定写到"本脚本所在目录"下的 output.okti
        os.path.dirname(os.path.abspath(__file__)),  # 本脚本所在目录
        OUTPUT_FILE_NAME)  # 固定的输出文件名

    print("正在读取：" + os.path.abspath(input_path))  # 提示用户正在读哪个文件，方便排查问题
    image = read_ppm(input_path)  # 读取并解析 PPM 图片
    print("图片尺寸：" + str(image["width"]) + " x " + str(image["height"]))  # 打印尺寸，便于确认读对了

    save_okti(image, output_path)  # 把图片数据编码后写入 OKTI 文件
    print("已写入：" + output_path)  # 提示用户写到了哪里


main()  # 程序入口：直接调用 main 函数开始运行
