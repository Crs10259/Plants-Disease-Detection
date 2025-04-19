import re

def contains_chinese_char(text):
    """Check if the text contains Chinese characters"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    return bool(chinese_pattern.search(text))

# Test cases
test_cases = [
    "测试.jpg",  # Chinese characters
    "测试test.jpg",  # Mixed Chinese and English
    "test.jpg",  # English only
    "test - 副本.jpg",  # Contains Chinese "副本" (copy)
    "test_picture_1.jpg",  # English with numbers and underscores
    "图片1.png"  # Chinese with numbers
]

# Write results to a file instead of printing to console
with open("chinese_detection_test_results.txt", "w", encoding="utf-8") as f:
    f.write("Testing Chinese character detection:\n")
    for test in test_cases:
        result = contains_chinese_char(test)
        f.write(f"'{test}': {'Contains Chinese' if result else 'No Chinese'}\n") 