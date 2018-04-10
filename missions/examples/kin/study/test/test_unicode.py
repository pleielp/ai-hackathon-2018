def print_ascii():
    for i in range(0, 128):
        print(chr(i), end=' ')
        if i % 16 == 15:
            print('')


def print_hangual():
    for i in range(0, 19 * 28 * 21):
        print(chr(44032 + i), end=' ')
        if i % 28 == 27:
            print('')
            if (i // 28) % 21 == 20:
                print('')


if __name__ == "__main__":
    print(print_ascii())
    # print(print_hangual())
