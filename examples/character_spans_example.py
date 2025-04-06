#!/usr/bin/env python3
"""
Example showing how to get character spans for segmented text.
"""

from charboundary import get_small_onnx_segmenter

def demonstrate_character_spans():
    """Demonstrate how to get character spans for segmented text."""
    
    # Get a pre-trained segmenter
    print("Loading pre-trained segmenter...")
    segmenter = get_small_onnx_segmenter()
    
    # Sample legal text
    legal_text = """
    The court in Brown v. Board of Education, 347 U.S. 483 (1954), declared that racial segregation 
    in public schools was unconstitutional. This landmark decision was delivered by Chief Justice 
    Earl Warren, who wrote the unanimous opinion. The ruling overturned the "separate but equal" 
    doctrine established in Plessy v. Ferguson, 163 U.S. 537 (1896).
    
    After the decision, implementation was delegated to district courts with orders to desegregate 
    "with all deliberate speed." The case was argued by NAACP attorney Thurgood Marshall, who later 
    became the first African American Supreme Court Justice.
    """
    
    print(f"\nOriginal text ({len(legal_text)} characters):\n{legal_text}")
    
    # 1. Basic sentence segmentation
    print("\n=== Basic sentence segmentation ===")
    sentences = segmenter.segment_to_sentences(legal_text)
    print(f"Found {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. {sentence}")
    
    # 2. Sentences with character spans
    print("\n=== Sentences with character spans ===")
    sentences_with_spans = segmenter.segment_to_sentences_with_spans(legal_text)
    print(f"Found {len(sentences_with_spans)} sentences with spans:")
    for i, (sentence, span) in enumerate(sentences_with_spans):
        start, end = span
        print(f"  {i+1}. ({start}, {end}): {sentence}")
    
    # 3. Just the character spans
    print("\n=== Just the character spans ===")
    spans = segmenter.get_sentence_spans(legal_text)
    print(f"Found {len(spans)} sentence spans:")
    for i, (start, end) in enumerate(spans):
        print(f"  {i+1}. Span ({start}, {end}) - Length: {end-start} chars")
    
    # Verify coverage (note: the model starts at position 5 after initial whitespace)
    total_coverage = sum(end - start for start, end in spans)
    print(f"\nTotal coverage: {total_coverage} of {len(legal_text)} characters")
    print(f"First span starts at position: {spans[0][0]}")
    print(f"Last span ends at position: {spans[-1][1]}")
    
    # The spans cover the entire text content (excluding leading whitespace)
    assert spans[-1][1] == len(legal_text), "Last span doesn't end at text length"
    assert spans[0][0] <= 5, "First span starts too far into the text"
    
    # 4. Paragraph spans
    print("\n=== Paragraph spans ===")
    paragraph_spans = segmenter.get_paragraph_spans(legal_text)
    print(f"Found {len(paragraph_spans)} paragraph spans:")
    for i, (start, end) in enumerate(paragraph_spans):
        excerpt = legal_text[start:min(start+60, end)] + ("..." if end-start > 60 else "")
        print(f"  {i+1}. Span ({start}, {end}) - Length: {end-start} chars")
        print(f"      Text: {excerpt}")
    
    # 5. Example of working with spans
    print("\n=== Working with spans (example) ===")
    print("Extracting legal citations using spans:")
    
    for i, (sentence, span) in enumerate(sentences_with_spans):
        # Simplified citation extraction (just for demonstration)
        if " v. " in sentence and "U.S." in sentence:
            start, end = span
            print(f"  Found citation in sentence {i+1}, chars {start}-{end}:")
            print(f"  {sentence}")
            print()

def main():
    """Run the example script."""
    print("CharBoundary Character Spans Example\n")
    demonstrate_character_spans()

if __name__ == "__main__":
    main()