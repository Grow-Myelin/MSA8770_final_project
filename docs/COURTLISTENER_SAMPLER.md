# CourtListener Corpus Sampler

A Python tool to sample 1000 documents from the CourtListener.com API for legal corpus creation.

## Overview

This project provides a flexible sampling tool for collecting court opinions from CourtListener.com, a comprehensive database of U.S. legal opinions. The tool supports multiple sampling strategies to ensure corpus diversity.

## Features

- **Multiple Sampling Strategies**:
  - **Sequential**: Simple pagination through results
  - **Stratified**: Temporal diversity by sampling across different time periods
  - **Diverse**: Jurisdictional diversity by sampling from different court types

- **Rate Limiting**: Respects CourtListener's 5,000 requests/hour limit
- **Multiple Output Formats**: JSON and JSONL
- **Metadata Tracking**: Automatically generates metadata about the corpus

## Prerequisites

1. **CourtListener API Token**: You need a free account at [courtlistener.com](https://www.courtlistener.com)
   - Sign up at: https://www.courtlistener.com/sign-in/register/
   - Get your API token from: https://www.courtlistener.com/api/rest-info/

2. **Python 3.7+**

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make the script executable (optional)
chmod +x courtlistener_sampler.py
```

## Usage

### Basic Usage

```bash
# Sequential sampling (simplest approach)
python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy sequential

# Stratified sampling (temporal diversity)
python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy stratified

# Diverse sampling (jurisdictional diversity)
python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy diverse
```

### Advanced Options

```bash
# Sample 500 documents instead of 1000
python courtlistener_sampler.py --token YOUR_API_TOKEN --count 500

# Use 30 years for stratified sampling
python courtlistener_sampler.py --token YOUR_API_TOKEN --strategy stratified --years 30

# Custom output directory
python courtlistener_sampler.py --token YOUR_API_TOKEN --output my_corpus
```

### Complete Example

```bash
python courtlistener_sampler.py \
  --token YOUR_API_TOKEN \
  --strategy diverse \
  --count 1000 \
  --output corpus
```

## Sampling Strategies Explained

### 1. Sequential Sampling
The simplest approach that collects documents in order as they appear in the API results.

**Pros:**
- Fast and straightforward
- Predictable

**Cons:**
- May lack diversity
- Results depend on API's default ordering

**Use when:** You need a quick corpus and diversity isn't critical

### 2. Stratified Sampling
Divides time into periods (e.g., 20 years) and samples evenly from each period.

**Pros:**
- Ensures temporal diversity
- Captures legal evolution over time
- Good for historical analysis

**Cons:**
- Slower (more API calls)
- May have fewer docs in older periods

**Use when:** You need temporal representation across years

### 3. Diverse Sampling
Samples from different court types to ensure jurisdictional diversity.

**Distribution:**
- 100 docs from Supreme Court (SCOTUS)
- 600 docs from Circuit Courts
- 300 docs from other courts

**Pros:**
- Jurisdictional diversity
- Balanced court representation
- Good for comparative analysis

**Cons:**
- Fixed distribution ratios

**Use when:** You need representation from different court levels

## Output Format

The tool creates a directory with three files:

### 1. `opinions.jsonl`
One JSON object per line (useful for streaming/processing):
```jsonl
{"id": 123, "court": "scotus", "case_name": "...", "text": "...", ...}
{"id": 124, "court": "ca9", "case_name": "...", "text": "...", ...}
```

### 2. `opinions.json`
Single JSON array (easier to read):
```json
[
  {
    "id": 123,
    "cluster_id": 456,
    "court": "scotus",
    "date_filed": "2023-01-15",
    "case_name": "Smith v. Jones",
    "text": "Full opinion text...",
    "url": "/opinion/123/smith-v-jones/"
  }
]
```

### 3. `metadata.json`
Corpus statistics:
```json
{
  "total_documents": 1000,
  "date_collected": "2025-11-05T12:00:00",
  "courts": ["scotus", "ca9", "ca1", ...],
  "date_range": {
    "earliest": "2005-01-01",
    "latest": "2025-11-05"
  }
}
```

## API Rate Limits

- **Authenticated users**: 5,000 requests/hour
- The tool implements automatic rate limiting (0.8 seconds between requests)
- Progress is displayed every 100 requests

**Estimated time for 1000 documents:**
- If 20 results per page: ~50 API calls × 0.8s = ~40 seconds
- Actual time varies based on API response time and pagination

## Data Structure

Each opinion document contains:

| Field | Description |
|-------|-------------|
| `id` | Unique opinion ID |
| `cluster_id` | Opinion cluster (group of related opinions) |
| `court` | Court identifier (e.g., "scotus", "ca9") |
| `date_filed` | Date the opinion was filed |
| `case_name` | Name of the case |
| `text` | Full text of the opinion |
| `url` | Relative URL on CourtListener |

## Troubleshooting

### "Authentication failed"
- Verify your API token is correct
- Check that you're using the format: `--token YOUR_TOKEN` (not with quotes)

### "Rate limit exceeded"
- Wait an hour for the limit to reset
- The tool already implements rate limiting, but if you ran other scripts, you may have hit the limit

### "No documents sampled"
- Check your internet connection
- Verify the API is accessible: https://www.courtlistener.com/api/rest/v4/
- Try a different sampling strategy

### "SSL Certificate Error"
- Update your Python SSL certificates
- Or add `--no-verify-ssl` flag (not recommended for production)

## CourtListener API Reference

- **API Documentation**: https://www.courtlistener.com/help/api/rest/
- **Search API**: https://www.courtlistener.com/help/api/rest/search/
- **Rate Limits**: https://www.courtlistener.com/help/api/
- **Register Account**: https://www.courtlistener.com/sign-in/register/

## Examples

### Example 1: Quick 100-document test corpus
```bash
python courtlistener_sampler.py --token YOUR_TOKEN --count 100 --output test_corpus
```

### Example 2: Large diverse corpus
```bash
python courtlistener_sampler.py --token YOUR_TOKEN --count 5000 --strategy diverse --output large_corpus
```

### Example 3: Historical analysis corpus (50 years)
```bash
python courtlistener_sampler.py --token YOUR_TOKEN --strategy stratified --years 50
```

## License

This project is for educational and research purposes. Please comply with CourtListener's Terms of Service when using their API.

## Contributing

Contributions welcome! Feel free to:
- Add new sampling strategies
- Improve error handling
- Add additional output formats
- Enhance documentation

## Citation

If you use this tool in research, please cite CourtListener:

> Free Law Project. (2025). CourtListener. https://www.courtlistener.com

## Support

For issues with:
- **This tool**: Open an issue in this repository
- **CourtListener API**: Contact Free Law Project at https://free.law/contact/
- **API access**: Check https://www.courtlistener.com/help/api/
