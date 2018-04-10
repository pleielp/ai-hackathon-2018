import tensorflow as tf

# special_chr_dict = {' ': 0, '!': 1, '"': 2, '#': 3, ... }
special_chr_list = [chr(i) for i in list(range(32, 48)) + list(range(58, 65)) + list(range(91, 97)) + list(range(123, 128)) + [ord('♡'), ord('♥'), ord('★'), ord('☆')]]
special_chr_dict = {n: i for i, n in enumerate(special_chr_list)}


def str_as_one_hot(str):
    return tf.one_hot(list(ord(x) - 44032 for x in str), 11072)


def vectorize_chr(char):
    # input: '각', output: [1, 1, 1, 0, 0, 0]
    # input: 'A', output: [0, 0, 0, 1, 0, 0]
    # input: '0', output: [0, 0, 0, 0, 1, 0]
    # input: ' ', output: [0, 0, 0, 0, 0, 1]
    num = ord(char)
    vector = []                 # [초성, 중성, 종성, 영어, 숫자, 특수문자]
    if ord('가') <= num <= ord('힣'):
        num_k = num - 44032
        z = num_k % 28              # jong
        y = (num_k // 28) % 21      # jung
        x = (num_k // 28) // 21     # cho
        vector = [x + 1, y + 1, z, 0, 0, 0]
    # if 31 < num < 128:              # ASCII
    elif 64 < num < 91:           # 영어 대문자
        vector = [0, 0, 0, num - 64, 0, 0]
    elif 96 < num < 123:        # 영어 소문자
        vector = [0, 0, 0, num - 96, 0, 0]
    elif 47 < num < 58:         # 숫자
        vector = [0, 0, 0, 0, num - 47, 0]
    else:                   # 특수문자
        try:
            vector = [0, 0, 0, 0, 0, special_chr_dict[char] + 1]
        except:  # no special character in special_chr_dict
            special_chr_dict[char] = len(special_chr_dict)
            vector = [0, 0, 0, 0, 0, special_chr_dict[char] + 1]
    return vector


def vectorize_str(str):
    # input: '각A0 '
    # output: [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    return [vectorize_chr(char) for char in str]

# for debugging
if __name__ == "__main__":
    print(str_as_one_hot('가나다'))
