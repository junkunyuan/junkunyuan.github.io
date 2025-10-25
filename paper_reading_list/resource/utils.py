"""
Utility functions for paper reading list HTML generation.

This module provides helper functions for venue name mapping,
color generation, and figure processing.
"""

import os
import re
from typing import Dict, Iterator, Generator

# Venue abbreviation to full name mapping
VENUE_NAME_DICT: Dict[str, str] = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "CVPR": "Conference on Computer Vision and Pattern Recognition",
    "ECAI": "European Conference on Artificial Intelligence",
    "ECCV": "European Conference on Computer Vision",
    "ICCV": "International Conference on Computer Vision",
    "ICASSP": "International Conference on Acoustics, Speech and Signal Processing",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "IJCV": "International Journal of Computer Vision",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "MM": "International Conference on Multimedia",
    "NeurIPS": "Advances in Neural Information Processing Systems",
    "SIGGRAPH-Asia": "ACM SIGGRAPH Annual Conference in Asia",
    "TKDD": "ACM Transactions on Knowledge Discovery from Data",
    "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
    "TMLR": "Transactions on Machine Learning Research",
    "TMM": "IEEE Transactions on Multimedia",
}

def get_venue_all(venue_abbr_date: str) -> str:
    """
    Get the full venue name from abbreviation.

    Args:
        venue_abbr_date: Venue abbreviation with optional date (e.g., "CVPR 2023")

    Returns:
        Full venue name or empty string if not found
    """
    venue_abbr = venue_abbr_date.split(" ", 1)[0]
    return VENUE_NAME_DICT.get(venue_abbr, "")


def border_color_generator() -> Generator[str, None, None]:
    """
    Generate border colors for paper sections in a cyclic manner.

    Yields:
        Color hex codes for styling paper sections
    """
    # colors = ["#ADDEFF", "#FFD6AD", "#B2EAC8", "#F0BBCC"]
    colors = ["#9ACAF0", "#F9D0A5", "#A0D0A0", "#F0BBCC", ]
    num_colors = len(colors)
    index = 0

    while True:
        yield colors[index % num_colors]
        index += 1

def convert_fig_cap_to_figure(text: str, name: str) -> str:
    """
    Convert figure markup to HTML figure elements.

    Processes text containing figure markup in the format:
    fig: image1.png 300 image2.png 200
    cap: Figure caption text

    Args:
        text: Text containing figure markup
        name: Paper name for constructing image paths

    Returns:
        HTML string with figure elements
    """
    # Split text to avoid processing code blocks
    parts = text.split("<pre>", 1)
    content_text = parts[0]
    code_block = ""
    if len(parts) == 2:
        code_block = "<pre>" + parts[1]

    lines = content_text.strip().splitlines()
    result_lines = []
    figure_count = 0
    line_index = 0

    # Build path prefix for images
    path_prefix = os.path.join("resource", "figs", name)

    while line_index < len(lines):
        line = lines[line_index].strip()

        if line.startswith("fig:"):
            figure_count += 1
            # Extract image information (filename width pairs)
            image_matches = re.findall(r"(\S+)\s+(\d+)", line[len("fig:"):])

            # Next line should be caption
            line_index += 1
            if line_index < len(lines):
                caption_line = lines[line_index].strip()
                if caption_line.startswith("cap:"):
                    caption = caption_line[len("cap:"):].strip()

                    # Build figure HTML
                    result_lines.append("<figure>")
                    for img_filename, width in image_matches:
                        img_path = os.path.join(path_prefix, f"{name}-{img_filename}")
                        result_lines.append(f"<img data-src='{img_path}' width={width}>")
                    result_lines.append("<figcaption>")
                    result_lines.append(f"<b>Figure {figure_count}.</b> {caption}")
                    result_lines.append("</figcaption>")
                    result_lines.append("</figure>")
                else:
                    # If next line is not a caption, treat current line as regular text
                    result_lines.append(line)
                    line_index -= 1  # Process the caption line again as regular text
            else:
                # No caption line, treat as regular text
                result_lines.append(line)
        else:
            result_lines.append(line)

        line_index += 1

    return "\n".join(result_lines) + code_block


# HTML template constants
TOP_BUTTON: str = """
<button id="backToTop" title="back to top">↑</button>
<script>
    const button = document.getElementById("backToTop");
    window.addEventListener("scroll", () => {
        if (document.documentElement.scrollTop > 300) {
            button.style.display = "block";
        } else {
            button.style.display = "none";
        }
    });

    function updateButtonPosition() {
        const bodyRect = document.body.getBoundingClientRect();
        const windowWidth = window.innerWidth;
        const rightOffset = Math.max((windowWidth - bodyRect.width) / 2, 10);
        button.style.right = rightOffset + "px";
    }

    window.addEventListener("resize", updateButtonPosition);
    window.addEventListener("load", updateButtonPosition);

    button.addEventListener("click", () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
</script>
"""

COPY_BUTTON: str = """
<script>
    function copy(elementId) {
        const codeElement = document.getElementById(elementId);
        const range = document.createRange();
        range.selectNode(codeElement);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);

        try {
            const successful = document.execCommand('copy');
            const btn = event.target;
            if (successful) {
                btn.textContent = '已复制!';
                setTimeout(() => {
                    btn.textContent = '复制代码';
                }, 2000);
            }
        } catch (err) {
            console.error('复制失败:', err);
        }

        window.getSelection().removeAllRanges();
    }
</script>
"""