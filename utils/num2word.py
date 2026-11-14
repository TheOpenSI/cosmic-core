NUM2WORD_DICT: dict[int, str] = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
    15: "Fifteen",
    16: "Sixteen",
    17: "Seventeen",
    18: "Eighteen",
    19: "Nineteen",
    20: "Twenty",
    30: "Thirty",
    40: "Forty",
    50: "Fifty",
    60: "Sixty",
    70: "Seventy",
    80: "Eighty",
    90: "Ninety",
    0: "Zero"
}


def convert_number2word(
    number: int
) -> str | None:
    """
    Convert number to letter.

    Args:
        number (int): numerical digits.

    Returns:
        str_num (str, optional): letter-style number, None if number not found in
                                 our dict.
    """
    # Cannot cover all numbers, if not in the dictionary, just return the input
    if number in NUM2WORD_DICT.keys():
        str_num: str = NUM2WORD_DICT[number].lower()
        return str_num

    else:
        return None