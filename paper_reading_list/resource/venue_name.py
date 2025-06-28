venue_name_dict = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "arXiv": "",
    "ICCV": "International Conference on Computer Vision",
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