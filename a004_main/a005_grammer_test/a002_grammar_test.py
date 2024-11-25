import pprint
import warnings


def unpack():
    lst = [1, 2, 3, 4]
    # v1 = *lst
    print(*lst)  # 1 2 3 4

    one, *others, four = lst  # *的作用是收集为列表
    print(others)  # one = 1, four = 4, others = [2, 3]

    three_d_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    (*two_d_lists,) = three_d_list


if __name__ == "__main__":
    from rich.console import Console
    from rich.pretty import Pretty

    # Your dictionary
    my_dict = {
        "key1": "value1",
        "key2": {
            "nested_key1": 123,
            "nested_key2": [1, 2, 3],
        },
        "key3": True,
    }

    # Create a console object
    console = Console(record=True)

    # Use Pretty to format the dictionary
    pretty_dict = Pretty(my_dict, expand_all=True)

    # Capture the output to a string
    console.print(pretty_dict)
    formatted_string = console.export_text()

    # Display the formatted string (optional)
    print(formatted_string)
