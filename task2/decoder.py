# decoder.py
# 姓名：侯程罡    学号：202600192
# 程序简介：把 OKTI 文本格式的图片解码回像素，并输出成 PPM（P6 二进制）文件。
#           输出文件名 = 原文件名（去掉扩展名）+ "_decoded.ppm"。
#           遇到非法文件时会打印一条说明信息，然后结束程序（共处理 4 类错误）。

import os  # 用于取出文件名、拼接路径
import sys  # 用于读取命令行参数

# ===== 常量（永不改变的值，放成常量可以避免"魔术数字"）=====
MAGIC_TOKEN = "okti"  # OKTI 文件第 1 行固定的内容
PPM_MAGIC_NUMBER = b"P6"  # PPM 二进制格式的标识，写在输出文件的开头
MAX_COLOR_VALUE = 255  # 每个颜色分量的最大值，写进 PPM 头部
COLOR_COMPONENT_COUNT = 3  # 每个像素由红、绿、蓝 3 个分量组成
DIFFERENCE_BIAS = 8  # 差值编码的偏移量：写出来的数字减 8 才是真实差值
MAX_SHORT_INDEX = 15  # 小写 i 只表示 0~15 的下标
MAX_COLOR_TABLE_SIZE = 256  # 颜色表最多保存 256 种颜色
INITIAL_COLOR = (0, 0, 0)  # 颜色表的初始内容与"上一个像素"的初始值：纯黑
OUTPUT_SUFFIX = "_decoded.ppm"  # 输出文件名要加的后缀
FULL_PIXEL_CHAR = "p"  # 完整 RGB 像素的标记字符
DIFFERENCE_CHAR = "d"  # 差值编码的标记字符
SHORT_RUN_CHAR = "r"  # 短 run 的标记字符
LONG_RUN_CHAR = "R"  # 长 run 的标记字符
SHORT_INDEX_CHAR = "i"  # 短索引的标记字符
LONG_INDEX_CHAR = "I"  # 长索引的标记字符
LEGAL_TOKEN_TYPES = (  # 所有合法的像素类型标记（用于检查非法类型）
    FULL_PIXEL_CHAR,  # p
    DIFFERENCE_CHAR,  # d
    SHORT_RUN_CHAR,  # r
    LONG_RUN_CHAR,  # R
    SHORT_INDEX_CHAR,  # i
    LONG_INDEX_CHAR,  # I
)


def fail(message):
    """打印一条错误信息并结束程序。

    参数 message：要显示给用户的错误说明。
    返回：这个函数不会返回（结束整个程序）。
    """
    print(message)  # 先打印错误说明
    sys.exit(1)  # 用非零状态码退出，表示程序是异常结束的


def check_token_type(line, token):
    """检查一段编码的类型字符是否合法，不合法就报错并结束程序。

    参数 line：这一行内容（用于在错误信息里原样显示）。
    参数 token：这一行去掉两端空白后的编码字符串。
    返回：None（检查通过时什么都不返回）。
    """
    kind = token[0]  # 首字符就是像素类型
    if kind not in LEGAL_TOKEN_TYPES:  # 不在 p/d/r/R/i/I 这六种之内
        fail("错误：第 " + str(line) + " 行出现了非法的像素类型 "
             + repr(kind) + "，该行内容是：" + repr(token))  # 报出类型、行号和这一行的内容


def read_okti_header(lines, start_index):
    """读取 OKTI 文件的头部（第 1 行 okti 与第 2 行的宽度和高度）。

    参数 lines：整个文件按行切好的列表。
    参数 start_index：从第几行开始读（通常是 0）。
    返回：一个二元组（图片信息字典, 像素编码开始的下一行下标）。
    """
    if lines[start_index] != MAGIC_TOKEN:  # 第 1 行必须是 okti
        fail("错误：这个文件不是 OKTI 文件，它的第 1 行是 "
             + repr(lines[start_index]) + "，而期望的是 " + repr(MAGIC_TOKEN) + "。")  # 显示实际读到的内容

    width_text, height_text = lines[start_index + 1].split()  # 第 2 行用空白隔开宽度和高度
    width = int(width_text)  # 转成整数
    height = int(height_text)  # 转成整数

    if width <= 0 or height <= 0:  # 宽度或高度不是正整数就说明尺寸非法
        fail("错误：图片尺寸非法，宽 = " + str(width) + "，高 = " + str(height)
             + "，两者都必须是大于 0 的正整数。")  # 报出实际读到的尺寸

    return {"width": width, "height": height}, start_index + 2  # 返回图片信息和编码起始行号


def read_okti_pixels(path):
    """读取一个 OKTI 文件并解出全部像素。

    参数 path：要读取的 .okti 文件路径。
    返回：一个字典，含 width（宽）、height（高）和 pixels（像素列表，
          每个像素是 (红, 绿, 蓝) 三个 0~255 的整数）。
    """
    if not os.path.exists(path):  # 先检查文件是否存在（错误处理 1）
        fail("错误：找不到文件 " + repr(path) + "，请检查文件名和路径是否正确。")  # 信息里带上文件名

    try:  # 打开文件也可能因为其他原因失败（例如没有读取权限）
        with open(path, "r", encoding="ascii") as okti_file:  # OKTI 是纯文本文件，用文本方式读
            lines = [line.rstrip("\n") for line in okti_file]  # 逐行读入并去掉行尾换行
    except OSError:  # 文件存在却打不开
        fail("错误：文件 " + repr(path) + " 存在，但无法打开读取。")  # 同样报出文件名

    image, next_line = read_okti_header(lines, 0)  # 先读头部两行
    pixels = []  # 用来按顺序存放解出来的像素
    color_table = [INITIAL_COLOR]  # 颜色表，按 OKTI 规定初始只含纯黑
    previous = INITIAL_COLOR  # "上一个像素"，按 OKTI 规定初始为纯黑
    wanted = image["width"] * image["height"]  # 一共需要多少个像素

    line_number = next_line  # 从编码第一行开始逐行处理
    while line_number < len(lines) and len(pixels) < wanted:  # 解够宽 × 高 个像素就停
        token = lines[line_number].strip()  # 取出这一行的编码（去掉两端空白）
        if token:  # 跳过空行（允许文件里有空行）
            check_token_type(line_number + 1, token)  # 先检查像素类型是否合法（错误处理 4），行号从 1 开始数
            color, copies = decode_token(token, previous, color_table)  # 解出颜色和这段代表几个像素
            for _ in range(copies):  # 把这段编码代表的像素逐个补进结果
                pixels.append(color)  # 重复的像素颜色相同，所以循环追加同一个颜色
            if color not in color_table:  # 新颜色要插到颜色表最前面
                if len(color_table) >= MAX_COLOR_TABLE_SIZE:  # 再插就会超过 256 项
                    color_table.pop()  # 先丢掉最后一项（最久没出现的颜色）
                color_table.insert(0, color)  # 插到表头
            previous = color  # 当前像素成为下一个像素的"上一个像素"
        line_number = line_number + 1  # 处理下一行

    image["pixels"] = pixels  # 把解出来的像素存进结果
    return image  # 返回图片数据


def decode_token(token, previous, color_table):
    """解码一段 OKTI 编码，算出它代表的颜色和像素个数。

    参数 token：一段编码字符串，例如 pff0000、d08f、r5、R20、i0、I11。
    参数 previous：上一个像素，形如 (红, 绿, 蓝)。
    参数 color_table：颜色表（列表），最新的颜色在最前面。
    返回：一个二元组（这段编码表示的颜色, 这段编码表示几个像素）。
    """
    kind = token[0]  # 首字符决定这段编码的类型

    if kind == FULL_PIXEL_CHAR:  # 完整 RGB：p + 6 位十六进制
        red = int(token[1:3], 16)  # 前两位是红
        green = int(token[3:5], 16)  # 中间两位是绿
        blue = int(token[5:7], 16)  # 最后两位是蓝
        return (red, green, blue), 1  # 表示 1 个像素

    if kind == DIFFERENCE_CHAR:  # 差值：d + 3 位十六进制，每位减 8 得到真实差值
        red_difference = int(token[1], 16) - DIFFERENCE_BIAS  # 红差
        green_difference = int(token[2], 16) - DIFFERENCE_BIAS  # 绿差
        blue_difference = int(token[3], 16) - DIFFERENCE_BIAS  # 蓝差
        red = previous[0] + red_difference  # 上一个像素的红加差值
        green = previous[1] + green_difference  # 绿同理
        blue = previous[2] + blue_difference  # 蓝同理
        return (red, green, blue), 1  # 表示 1 个像素

    if kind == SHORT_RUN_CHAR or kind == LONG_RUN_CHAR:  # run：r/R 表示再复制 N 个上一个像素
        copies = int(token[1:], 16)  # 后面的十六进制数字就是"再复制几个"
        return previous, copies  # 颜色等于上一个像素，个数就是 N

    if kind == SHORT_INDEX_CHAR or kind == LONG_INDEX_CHAR:  # 颜色表索引：i/I + 下标
        position = int(token[1:], 16)  # 下标是十六进制的
        return color_table[position], 1  # 从颜色表里取出对应颜色，表示 1 个像素

    raise ValueError("无法识别的编码：" + token)  # 合法文件里不该走到这里


def output_path_for(input_path):
    """根据输入的 .okti 路径算出输出 .ppm 的路径。

    参数 input_path：输入的 .okti 文件路径。
    返回：输出文件的路径（与原文件同目录，文件名去掉扩展名后加 _decoded.ppm）。
    """
    directory = os.path.dirname(os.path.abspath(input_path))  # 取输入文件所在目录
    file_name = os.path.basename(input_path)  # 取文件名（含扩展名）
    base_name = os.path.splitext(file_name)[0]  # 去掉扩展名，只留主名
    return os.path.join(directory, base_name + OUTPUT_SUFFIX)  # 拼出输出路径


def write_ppm(image, path):
    """把像素写成 PPM（P6 二进制）文件。

    参数 image：含 width、height、pixels 的图片字典。
    参数 path：要写入的 .ppm 文件路径。
    返回：None（这个函数只负责写文件）。
    """
    header = b"%s\n%d %d\n%d\n" % (  # PPM 头部：标识、宽、高、颜色最大值
        PPM_MAGIC_NUMBER,  # P6
        image["width"],  # 宽度
        image["height"],  # 高度
        MAX_COLOR_VALUE)  # 255

    pixel_bytes = bytearray()  # 用来逐个堆放像素字节
    for red, green, blue in image["pixels"]:  # 按顺序取出每个像素
        pixel_bytes.append(red)  # 红
        pixel_bytes.append(green)  # 绿
        pixel_bytes.append(blue)  # 蓝

    with open(path, "wb") as ppm_file:  # 以二进制方式写入，因为像素不是文字
        ppm_file.write(header)  # 先写头部
        ppm_file.write(pixel_bytes)  # 再写像素数据


def main():
    """主程序：从命令行或用户输入取得文件名，解码并写出 PPM 文件。

    参数：无（输入文件名来自命令行参数或用户输入）。
    返回：None。
    """
    if len(sys.argv) == 2:  # 命令行恰好给了 1 个参数
        input_path = sys.argv[1]  # 就用它当要解码的 OKTI 文件名
    elif len(sys.argv) == 1:  # 没有给任何命令行参数
        input_path = input("请输入要解码的 OKTI 文件名：")  # 用 input 函数问用户要文件名
    else:  # 给了 2 个或更多参数
        print("错误：命令行参数太多，最多只能给一个 OKTI 文件名。")
        sys.exit(1)  # 结束程序，并用非零状态码表示是异常结束

    image = read_okti_pixels(input_path)  # 读入并解码
    print("解码完成：%d x %d，共 %d 个像素" % (
        image["width"], image["height"], len(image["pixels"])))  # 报告结果

    output_path = output_path_for(input_path)  # 算出输出文件名
    write_ppm(image, output_path)  # 写出 PPM 文件
    print("已写入：" + output_path)  # 提示输出位置


main()  # 程序入口：调用 main 函数开始运行
