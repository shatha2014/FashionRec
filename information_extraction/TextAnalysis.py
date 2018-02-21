from Preprocessor import PreProcessor
from InformationExtraction import InformationExtractor
import os
import json
import time
from collections import Counter

"""
Script for analyzing users, uses Preprocessor.py and InformationExtraction.py
Non-distributed, single threaded.
"""

def read_corpora(corpora_file_path):
    """ Read Corpora"""
    ids = []
    comments = []
    captions = []
    tags = []
    with open(corpora_file_path, "r") as corpora_file:
        for line in corpora_file:
            parts = line.split(",")
            if len(parts) == 4:
                ids.append(parts[0])
                comments.append(parts[1])
                captions.append(parts[2])
                tags.append(parts[3])
    return ids, comments, captions, tags

def read_gazetter(file_path):
    """ Read Domain Knowledge Files"""
    words = []
    with open(file_path, "r") as file:
        for line in file:
            words.append(line.replace("\n", "").lower())
    return words

def filterOccurenceCount(iter):
    """ Filter Occurence Counts """
    A = Counter(iter)
    return {x : A[x] for x in A if A[x] >= 3}.keys()

def filterEmojis(emojis, tokens):
    """ Filter Emojis"""
    return filter(lambda x: x not in emojis, tokens)

def analyze_user(user, information_extractor):
    """ Analyze user with semantic/syntactic similarity and Probase"""
    start = time.time()
    print("Analyzing user {0}".format(user))
    ids, comments, captions, tags = read_corpora("./data/small.csv")
    print("number of posts: {0}\n{1}".format(len(ids), len(captions)))
    normalized_text = PreProcessor(ids, comments, captions, tags)
    user_dict = {}
    user_dict["name"] = user
    analyzed_posts = []
    for i in range(len(normalized_text.tokens_all)):
        if i % 10 == 0:
            print("Processing post {0} out of {1}".format(i, len(normalized_text.tokens_all)))
        post_dict = {}
        like_to_know_it_links = sorted(information_extractor.get_liketoknowitlinks(normalized_text.tokens_all[i]), reverse=True, key=lambda x: x[1])
        tokens = filterOccurenceCount(normalized_text.tokens_all[i])
        tokens = filterEmojis(normalized_text.emojis[i], tokens)
        styles = sorted(list(set(information_extractor.find_closest_semantic(tokens, 5, information_extractor.styles_lemmas.keys()) + information_extractor.find_closest_syntactic(normalized_text.tokens_all[i], 5, information_extractor.styles_lemmas.keys()))), reverse=True, key=lambda x: x[1])
        items = sorted(list(set(information_extractor.find_closest_semantic(tokens, 5, information_extractor.items_lemmas.keys()) + information_extractor.find_closest_semantic(normalized_text.tokens_all[i], 5, information_extractor.items_lemmas.keys()))),reverse=True, key=lambda x: x[1])
        materials = sorted(list(set(information_extractor.find_closest_semantic(tokens, 2, information_extractor.materials_lemmas.keys()) + information_extractor.find_closest_syntactic(normalized_text.tokens_all[i], 2, information_extractor.materials_lemmas.keys()))),reverse=True, key=lambda x: x[1])
        brands = sorted(list(set(information_extractor.find_closest_syntactic(tokens, 4, information_extractor.companies))),reverse=True, key=lambda x: x[1])
        colors = sorted(list(set(information_extractor.find_closest_semantic(tokens, 5, information_extractor.colors) + information_extractor.find_closest_syntactic(tokens, 5, information_extractor.colors))),reverse=True, key=lambda x: x[1])
        patterns = sorted(list(set(information_extractor.find_closest_semantic(tokens, 5, information_extractor.patterns) + information_extractor.find_closest_syntactic(tokens, 5, information_extractor.patterns))),reverse=True, key=lambda x: x[1])
        ranked_brands = sorted(re_rank_brands(information_extractor, brands),reverse=True, key=lambda x: x[1])
        ranked_materials = sorted(re_rank_materials(information_extractor, materials),reverse=True, key=lambda x: x[1])
        post_dict["styles"] = styles
        post_dict["items"] = items
        post_dict["brands"] = ranked_brands
        post_dict["colors"] = colors
        post_dict["patterns"] = patterns
        post_dict["materials"] = ranked_materials
        post_dict["liketkit"] = like_to_know_it_links
        post_dict["hashtags"] = normalized_text.hashtags[i]
        analyzed_posts.append(post_dict)
        end = time.time()
        print(end - start)

    user_dict["posts"] = analyzed_posts
    output_path = "analyzed_users/" + user + "/" + user + ".json"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as fp:
        json.dump(user_dict, fp)


def re_rank_materials(information_extractor, materials):
    """ Re-rank materials based on probase lookups"""
    ranked_materials = map(lambda material: material_rank_mapper(information_extractor, material), materials)
    return ranked_materials

def material_rank_mapper(information_extractor, material_):
    """ Re-rank materials based on probase lookups"""
    material, rank = material_
    factor_probase = lookup_material_probase(information_extractor, material, 10)
    return (material, factor_probase*rank)

def lookup_material_probase(information_extractor, query, num):
    """Lookup material in probase"""
    company_params = {
        'instance': query,
        'topK': num
    }
    result = information_extractor.lookup_probase(company_params)
    rank = information_extractor.rank_probase_result_company(result)
    return rank

def re_rank_brands(information_extractor, brands):
    """ Re-rank brands based on probase lookups"""
    ranked_brands = map(lambda brand: brand_rank_mapper(information_extractor, brand), brands)
    return ranked_brands

def brand_rank_mapper(information_extractor, brand_):
    """ Re-rank brands based on probase lookups"""
    brand, rank = brand_
    factor_probase = lookup_company_probase(information_extractor, brand, 10)
    return (brand, rank*factor_probase)

def lookup_company_probase(information_extractor, query, num):
    """ Lookup company in probase"""
    company_params = {
        'instance': query,
        'topK': num
    }
    result = information_extractor.lookup_probase(company_params)
    rank = information_extractor.rank_probase_result_company(result)
    return rank

def lookup_company_google(information_extractor, query, num):
    """ Lookup company in google search"""
    company_params = {
        'query': query,
        'limit': num,
        'languages': "en",
        'indent': False,
        'key': InformationExtractor.api_key,
        'types': "Organization"
    }
    result = information_extractor.lookup_google(company_params)
    rank = information_extractor.rank_google_result_company(result)
    return rank


def main():
    """ Program entrypoint, orchestrates the pipeline """
    materials = read_gazetter("./domain_data/material.csv")
    items = read_gazetter("./domain_data/items.csv")
    styles = read_gazetter("./domain_data/general_style.csv")
    companies = read_gazetter("./domain_data/brands.csv")
    probase_brands = read_gazetter("./domain_data/probase_brands.csv")
    probase_materials = read_gazetter("./domain_data/probase_materials.csv")
    brands_keywords_google_search = read_gazetter("./domain_data/brands_keywords_google_search.csv")
    materials_keywords_google_search = read_gazetter("./domain_data/materials_keywords_google_search.csv")
    colors = read_gazetter("./domain_data/colors.csv")
    patterns = read_gazetter("./domain_data/patterns.csv")
    information_extractor = InformationExtractor("./vectors/vectors.vec", companies, styles, materials, items,
                                                 brands_keywords_google_search, materials_keywords_google_search,
                                                 probase_brands, probase_materials, colors, patterns)
    users = os.listdir("data/ig/")
    print users
    analyze_user("test", information_extractor)


if __name__ == '__main__':
    main()

