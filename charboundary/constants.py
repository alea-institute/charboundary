"""
Constants used throughout the charboundary library.
"""

# Punctuation list, including unicode
PUNCTUATION_CHAR_LIST = frozenset([
    ')', ']', '}', 
    '\u0f3b', '\u0f3d', '\u169c',  # Unicode brackets and close quotes
    '\u2046', '\u207e', '\u208e', '\u2309', '\u230b', 
    '\u3009', '\u2769', '\u276b', '\u276d', '\u276f', 
    '\u2771', '\u2773', '\u2775', '\u27c6', '\u27e7', 
    '\u27e9', '\u27eb', '\u27ed', '\u27ef', '\u2984', 
    '\u2986', '\u2988', '\u298a', '\u298c', '\u298e', 
    '\u2990', '\u2992', '\u2994', '\u2996', '\u2998', 
    '\u29d9', '\u29db', '\u29fd', '\u2e23', '\u2e25', 
    '\u2e27', '\u2e29', '\u2e56', '\u2e58', '\u2e5a', 
    '\u2e5c', '\u3009', '\u300b', '\u300d', '\u300f',
    '\u3011', '\u3015', '\u3017', '\u3019', '\u301b', 
    '\u301e', '\u301f', '\ufd3e', '\ufe18', '\ufe36', 
    '\ufe38', '\ufe3a', '\ufe3c', '\ufe3e', '\ufe40', 
    '\ufe42', '\ufe44', '\ufe48', '\ufe5a', '\ufe5c', 
    '\ufe5e', '\uff09', '\uff3d', '\uff5d', '\uff60', 
    '\uff63', '.', ')', '\u00bb', '\u2019', '\u201d', 
    '\u203a', '\u2e03', '\u2e05', '\u2e0a', '\u2e0d', 
    '\u2e1d', '\u2e21', '\u201c', '\u201d', '_', 
    '\u203f', '\u2040', '\u2054', '\ufe33', '\ufe34', 
    '\ufe4d', '\ufe4e', '\ufe4f', '\uff3f', ':', ';', 
    ',', '&', '-', '\u05be', '\u05bf', '\u1400', 
    '\u1806', '\u2010', '\u2011', '\u2012', '\u2013', 
    '\u2014', '\u2015', '\u2e17', '\u2e1a', '\u2e3a', 
    '\u2e3b', '\u2e40', '\u2e5d', '\u301c', '\u3030',
    '\u30a0', '\ufe31', '\ufe32', '\ufe58', '\ufe63', 
    '\uff0d', '\U00010e2d', '\u002d', '\u2013', '\u2014'
])

# whitespace
WS_CHAR_LIST = frozenset([
    ' ', '\xa0', '\u1680', '\u2000', '\u2001', '\u2002',
    '\u2003', '\u2004', '\u2005', '\u2006', '\u2007',
    '\u2008', '\u2009', '\u200a', '\u202f', '\u205f',
    '\u3000', '\t', '\u2028', '\r', '\n'
])

# list of characters that can possibly end a sentence
TERMINAL_SENTENCE_CHAR_LIST = frozenset([
    # punctuation marks
    '.',  # period
    '!',  # exclamation mark
    '?',  # question mark
    ';',  # semicolon (often used in complex sentences)
    # quotations (straight and curly)
    '"',  # straight double quotes
    '\u201d',  # right double quotation mark (curly)
    "'",  # straight single quote
    '\u2019',  # right single quotation mark (curly)
    # other punctuation marks
    ':',  # colon (can end sentences in certain contexts)
    '...',  # ellipsis (can indicate a trailing off or incomplete thought)
])

# Primary terminators - more likely to end sentences
PRIMARY_TERMINATORS = frozenset(['.', '!', '?'])

# Secondary terminators - less likely to end sentences on their own
SECONDARY_TERMINATORS = frozenset(['"', '\u201d', "'", '\u2019', ';', ':'])

# Opening quotation marks
OPENING_QUOTES = frozenset(['"', '\u201c', "'", '\u2018'])

# Closing quotation marks
CLOSING_QUOTES = frozenset(['"', '\u201d', "'", '\u2019'])

# list of characters that can indicate the end of a paragraph
TERMINAL_PARAGRAPH_CHAR_LIST = frozenset([
    # characters that can end a paragraph
    '\n',  # newline character (common in text files)
    '\r',  # carriage return (used in some text formats)
    # terminal sentence characters can also indicate end of paragraph
    # if they appear at the end of a line
])

# Annotation tags
SENTENCE_TAG = "<|sentence|>"
PARAGRAPH_TAG = "<|paragraph|>"

# Default list of common abbreviations that end with a period but don't end sentences
DEFAULT_ABBREVIATIONS = []

# Legal-specific abbreviations
LEGAL_ABBREVIATIONS = []

# Enumeration patterns - used to detect list items
LIST_MARKERS = [
    '(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)',
    '(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)', '(vii)', '(viii)', '(ix)', '(x)',
    '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
    '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
    'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.',
    'i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.',
    '•', '·', '○', '●', '■', '□', '▪', '▫',
]

# Conjunction patterns that often appear in the last item of a list
LIST_CONJUNCTIONS = [
    ' and ',
    ' or ',
    ' and/or ',
    ' as well as ',
]

# Typical list introduction patterns
LIST_INTROS = [
    'following:',
    'as follows:',
    'include:',
    'including:',
    'such as:',
    'namely:',
    'listed below:',
    'items below:',
    'the following items:',
]
