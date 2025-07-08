import re
import os

venue_name_dict = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "arXiv": "arXiv preprint",
    "CVPR": "Conference on Computer Vision and Pattern Recognition",
    "ICCV": "International Conference on Computer Vision",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "IJCV": "International Journal of Computer Vision",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "MM": "International Conference on Multimedia",
    "NeurIPS": "Advances in Neural Information Processing Systems",
    "SIGGRAPH-Asia": "ACM SIGGRAPH Annual Conference in Asia",
    "TKDE": "IEEE Transactions on Knowledge and Data Engineering",
    "TMLR": "Transactions on Machine Learning Research",
}

def get_venue_all(venue_abbr_date):
    venue_abbr = venue_abbr_date.split(" ", 1)[0]
    venue_all = venue_name_dict.get(venue_abbr, "")
    
    return venue_all


def border_color_generator():
    colors = ["#ADDEFF","#FFD6AD", "#B2EEC8", "#FFBBCC"]
    num_color = len(colors)
    num = 0
    while True:
        yield colors[num % num_color]
        num += 1

def convert_fig_cap_to_figure(text, name):
    lines = text.strip().splitlines()
    result = []
    fig_count = 0
    i = 0
    path_prefix = f"resource/figs/{name}/{name}-"
    path_prefix = os.path.join("resource", "figs", name)

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("fig:"):
            fig_count += 1
            # Extract image info
            image_info = re.findall(r"(\S+)\s+(\d+)", line[len("fig:"):])
            
            # Next line should be caption
            i += 1
            cap_line = lines[i].strip()
            caption = cap_line[len("cap:"):].strip()
            
            # Build figure block
            result.append("<figure>")
            for img, width in image_info:
                img_path = os.path.join(path_prefix, f"{name}-{img}")
                result.append(f"<img data-src='{img_path}' width={width}>")
            result.append("<figcaption>")
            result.append(f"<b>Figure {fig_count}.</b> {caption}")
            result.append("</figcaption>")
            result.append("</figure>")
        else:
            result.append(line)
        i += 1
    result_content = "\n".join(result)

    # if "MiniGPT-v2" in result_content:
    #     import pdb; pdb.set_trace()

    return result_content