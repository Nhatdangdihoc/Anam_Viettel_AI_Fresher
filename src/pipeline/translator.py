"""
Translation Engine
Dịch thuật sử dụng Google Translate (deep_translator).
"""

import time
from typing import List


def translate_sentences(
    sentences: List[str],
    src: str = "vi",
    dest: str = "en",
    delay: float = 0.3,
) -> List[str]:
    """
    Dịch danh sách câu sang ngôn ngữ đích.

    Args:
        sentences: Danh sách câu cần dịch.
        src:       Ngôn ngữ nguồn (mặc định: vi).
        dest:      Ngôn ngữ đích (mặc định: en).
        delay:     Delay giữa mỗi request (tránh rate limit).

    Returns:
        Danh sách câu đã dịch.
    """
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=src, target=dest)
    translated = []

    total = len(sentences)
    for i, s in enumerate(sentences):
        try:
            result = translator.translate(s)
            translated.append(result if result else s)
            if delay > 0:
                time.sleep(delay)
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  [Translate] {i + 1}/{total} sentences done")
        except Exception as e:
            print(f"  [!] Translation error at sentence {i}: {e}")
            translated.append(f"[Translation Error] {s}")

    return translated
