import pytest
from triton_eval.agents.tools import extract_code

# Test cases: (input_string, expected_output)
test_cases = [
    # Basic cases
    ("no code here", ""),
    ("", ""),
    ("```python\nprint('hello')\n```", "print('hello')"),
    ("some text ```python\nprint('world')\n``` more text", "print('world')"),
    ("```\nraw code\n```", "raw code"),
    ("text ```\nother raw code\n``` end", "other raw code"),

    # Triton specific cases
    ("<triton>raw triton content</triton>", "raw triton content"),
    ("text <triton>  stripped raw triton  </triton> more", "stripped raw triton"),
    ("<triton>```python\nprint('triton_py')\n```</triton>", "print('triton_py')"),
    ("before <triton> ```python\nprint('mid_py')\n``` </triton> after", "print('mid_py')"),
    ("<triton>```\nraw_in_triton\n```</triton>", "raw_in_triton"),
    ("text <triton> ```\nraw_in_triton_2\n``` </triton> end", "raw_in_triton_2"),
    
    # Triton with mixed internal content (last code block wins)
    ("<triton>```\nfirst_raw\n``` ```python\nsecond_py\n```</triton>", "second_py"),
    ("<triton>```python\nfirst_py\n``` ```\nsecond_raw\n```</triton>", "second_raw"),
    ("<triton>text ```python\ninternal_py\n``` more text</triton>", "internal_py"),
    ("<triton>text ```\ninternal_raw\n``` more text</triton>", "internal_raw"),
    ("<triton> ```python\npy1\n``` text ```python\npy2\n``` </triton>", "py2"),
    ("<triton> ```\nraw1\n``` text ```\nraw2\n``` </triton>", "raw2"),

    # Multiple Triton blocks (last triton block wins)
    ("<triton>```python\nfirst_triton_py\n```</triton> text <triton>```python\nsecond_triton_py\n```</triton>", "second_triton_py"),
    ("<triton>```\nfirst_triton_raw\n```</triton> text <triton>```\nsecond_triton_raw\n```</triton>", "second_triton_raw"),
    ("<triton>raw_1</triton> blah <triton>raw_2_content</triton>", "raw_2_content"),
    ("<triton>```python\npy_in_first\n```</triton> text <triton>raw_in_second</triton>", "raw_in_second"),
    ("<triton>raw_in_first</triton> text <triton>```python\npy_in_second\n```</triton>", "py_in_second"),

    # Mixed Triton and standalone blocks (Triton has precedence)
    ("```python\nstandalone_py_first\n``` <triton>```python\ntriton_py_second\n```</triton>", "triton_py_second"),
    ("<triton>```python\ntriton_py_first\n```</triton> ```python\nstandalone_py_second\n```", "triton_py_first"), # Last triton content matters
    ("```\nstandalone_raw_first\n``` <triton>raw_triton_second</triton>", "raw_triton_second"),
    ("<triton>raw_triton_first</triton> ```\nstandalone_raw_second\n```", "raw_triton_first"),

    # Standalone multiple blocks (last wins)
    ("```python\nfirst_py\n``` ```python\nsecond_py\n```", "second_py"),
    ("```\nfirst_raw\n``` ```\nsecond_raw\n```", "second_raw"),
    ("```python\npy_code\n``` ```\nraw_code_after\n```", "raw_code_after"),
    ("```\nraw_code_before\n``` ```python\npy_code_after\n```", "py_code_after"),

    # Edge cases with ```python that might be caught by generic ```
    ("```python\nprint('test')\n```", "print('test')"), # Ensure python block isn't missed
    ("<triton>```python\nprint('triton_test')\n```</triton>", "print('triton_test')"),
    ("```\nnot python\n``` ```python\nis_python\n```", "is_python"),
    ("<triton>```\nnot_py_in_triton\n``` ```python\nis_py_in_triton\n```</triton>", "is_py_in_triton"),

    # Incomplete/Malformed cases - Triton's raw content fallback
    ("<triton>```python not_closed_block</triton>", "```python not_closed_block"),
    ("<triton>``` not_closed_raw</triton>", "``` not_closed_raw"),
    ("<triton> some content ```python \n partially_started", "some content ```python \n partially_started"),
    ("<triton> text with ```python marker but no block</triton>", "text with ```python marker but no block"),
    ("<triton> text with ``` marker but no block</triton>", "text with ``` marker but no block"),

    # More complex nesting and ordering
    ("text <triton> ```\nraw_in_T1\n``` </triton> mid <triton> ```python\npy_in_T2\n``` </triton> end", "py_in_T2"),
    ("text <triton> ```python\npy_in_T1\n``` </triton> mid <triton> ```\nraw_in_T2\n``` </triton> end", "raw_in_T2"),
    ("```python\nouter_py1\n``` <triton> <triton_content> ```python\ninner_py\n``` </triton_content> </triton> ```python\nouter_py2\n```", "inner_py"), # Triton is last "major" block
    ("```python\nouter_py1\n``` <triton> <triton_content> ```\ninner_raw\n``` </triton_content> </triton> ```\nouter_raw2\n```", "inner_raw"),

    # Case where generic block pattern might accidentally include "python" in its content
    ("```\npython = 1\nprint(python)\n```", "python = 1\nprint(python)"), # This is a generic block, not ```python
    ("<triton>```\npython_var = 'abc'\n```</triton>", "python_var = 'abc'"),
    
    # No actual code content, just the markers
    ("```python\n```", ""),
    ("```\n```", ""),
    ("<triton>```python\n```</triton>", ""),
    ("<triton>```\n```</triton>", ""),
    ("<triton></triton>", ""),

    # Whitespace variations
    ("```python\n\n\n   code   \n\n\n```", "code"),
    ("<triton>   ```python   \n  py_code  \n  ```   </triton>", "py_code"),
    ("<triton>  some raw  </triton>", "some raw"),
]

# Insert new multi-line tests here
multi_line_test_cases = [
    # Multi-line and indented code
    (
        "```python\ndef foo():\n    print('hello')\n    return 42\n```",
        "def foo():\n    print('hello')\n    return 42"
    ),
    (
        "<triton>```python\nclass Bar:\n  def __init__(self):\n    self.x = 10\n```</triton>",
        "class Bar:\n  def __init__(self):\n    self.x = 10"
    ),
    (
        "text then\n<triton>\n  ```python\n  # A comment\n  def complex_func(a, b):\n    if a > b:\n      return a - b\n    else:\n      return b - a\n  ```\n</triton>\nand after",
        "# A comment\ndef complex_func(a, b):\n  if a > b:\n    return a - b\n  else:\n    return b - a"
    ),
    (
        "```python\n# Just comments\n# and spaces\n\n\n```",
        "# Just comments\n# and spaces"
    ),
    (
        "<triton>```\n# Raw block with indent\n  line1\n    line2\n```</triton>",
        "# Raw block with indent\n  line1\n    line2"
    ),
     # Multi-line inside triton, triton content is last
    (
        "<triton>```python\ndef outer_func():\n  pass\n```\nSome text\n```python\ndef inner_func():\n  # This is the one\n  x = 1\n  y = 2\n  return x + y\n```\n</triton>",
        "def inner_func():\n  # This is the one\n  x = 1\n  y = 2\n  return x + y"
    ),
    # Standalone multi-line, last one wins
    (
        "```python\nprint('first')\n```\n\n```python\ndef last_one():\n  val = 'correct'\n  return val\n```",
        "def last_one():\n  val = 'correct'\n  return val"
    )
]

test_cases.extend(multi_line_test_cases)

@pytest.mark.parametrize("input_string, expected_output", test_cases)
def test_extract_code(input_string, expected_output):
    assert extract_code(input_string) == expected_output

# Example of how to run this with pytest:
# Ensure pytest is installed: pip install pytest
# Navigate to the directory containing axolotl_dev and run: pytest
