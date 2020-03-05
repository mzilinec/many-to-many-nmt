import os
from tqdm.auto import tqdm


def create_data_config(training_files, testing_files, languages):
    return {"training_files": training_files, "testing_files": testing_files, "languages": languages}


def add_multilingual_files(input_dir, prefix, dataset):
    """
    Adds all files from the input_dir to the specified dataset.
    The files are expected to contain pairs of parallel sentences, following this naming convention:
    {prefix}.{src}-{tgt}.({src}|{tgt}),
    where {src} denotes the source language, {tgt} denotes the target language,
    and {src}-{tgt} denotes the language pair to which this file belongs.
    Example: "OpenSubtitles.cs-en.en"
    
    :returns: dataset, set_of_found_languages
    """
    files = os.listdir(input_dir)
    language_pairs = set([f.split(".")[-2] for f in files if f.startswith(prefix)])
    languages = set([lang for pair in language_pairs for lang in pair.split("-")])
    for pair in language_pairs:
        lang1, lang2 = pair.split("-")
        file1 = os.path.join(input_dir, f"{prefix}.{lang1}-{lang2}.{lang1}")
        file2 = os.path.join(input_dir, f"{prefix}.{lang1}-{lang2}.{lang2}")
        # append source and target files in both directions
        dataset.append(["http://example.com/", (os.path.abspath(file1), os.path.abspath(file2), None)])
    return dataset, languages


def add_mixed_files(source_file, target_file, index_file, dataset, delimiter=" "):
    """
    Adds a parallel corpus containing mixed language pairs to the specified dataset.
    :param source_file: The file containing the source sentences.
    :param target_file: The file containing the target sentences.
    :param index_file: A file containing the appropriate language codes for each source-target sentence pair.
                       Example: en-cs\nen-de\nen-fr
    :param delimiter: The delimiter that was used in index_file, such as "-" or " ".
    """
    languages = set()
    with open(index_file, "r") as fp:
        for line in tqdm(fp):
            lang1, lang2 = line.rstrip().split(delimiter)
            languages.add(lang1)
            languages.add(lang2)
    
    dataset.append(["http://example.com/", 
        (os.path.abspath(source_file), os.path.abspath(target_file), os.path.abspath(index_file))])
    return dataset, languages


def add_prefixed_files(source_file, target_file, dataset):
    """Don't use this."""
    prefixes = set()
#     with open(source_file, "r") as fp:
#         for line in tqdm(fp):
#             prefix = line.split(" ", maxsplit=1)[0]
#             prefixes.add(prefix)
    dataset.append(["http://example.com/", (source_file, target_file, 42)])
    return dataset, prefixes
