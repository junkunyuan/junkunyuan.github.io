venue_name_dict = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "arXiv": "",
    "ICCV": "International Conference on Computer Vision",
    "ICLR": "International Conference on Learning Representations",
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