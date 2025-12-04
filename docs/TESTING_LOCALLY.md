# Testing the Sampler Locally

Due to CourtListener's bot protection (Cloudflare), the sampler cannot be tested from cloud/container environments. You'll need to run it locally on your machine.

## Why Cloud Testing Fails

CourtListener.com blocks requests from:
- Data center IP addresses
- Cloud containers
- Automated environments without proper browser signatures

This is **not** a bug in the code - it's intentional bot protection on their end.

## How to Test on Your Local Machine

### Option 1: Use Environment Variable (Recommended)

```bash
# Set the environment variable
export COURTLISTENER_TOKEN='7eeb46ac3e491aff751330d8004e99000dc8c85f'

# Run the sampler
python courtlistener_sampler.py --count 20 --output test_corpus
```

### Option 2: Use Config File (Most Secure)

1. Create a `config.json` file:
```json
{
  "api_token": "7eeb46ac3e491aff751330d8004e99000dc8c85f"
}
```

2. Run with config:
```bash
python courtlistener_sampler.py --config config.json --count 20 --output test_corpus
```

**Note:** `config.json` is in `.gitignore` so it won't be committed to git.

### Option 3: Pass Token Directly

```bash
python courtlistener_sampler.py --token 7eeb46ac3e491aff751330d8004e99000dc8c85f --count 20
```

## Quick Test (20 documents)

Start with a small test to verify everything works:

```bash
# Using environment variable
export COURTLISTENER_TOKEN='7eeb46ac3e491aff751330d8004e99000dc8c85f'
python courtlistener_sampler.py --count 20 --output test_corpus --strategy sequential
```

Expected output:
```
Using sequential sampling strategy...
Starting sequential sampling for 20 documents...
Collected 20/20 documents...

Corpus saved successfully!
  - Documents: 20
  - JSONL: test_corpus/opinions.jsonl
  - JSON: test_corpus/opinions.json
  - Metadata: test_corpus/metadata.json
  - Total API requests: 2

✓ Successfully sampled 20 documents!
```

## Full 1000 Document Run

Once the test works, run the full corpus:

```bash
# Sequential (fastest)
python courtlistener_sampler.py --count 1000 --strategy sequential

# Stratified (temporal diversity)
python courtlistener_sampler.py --count 1000 --strategy stratified --years 20

# Diverse (jurisdictional diversity) - RECOMMENDED
python courtlistener_sampler.py --count 1000 --strategy diverse
```

## Expected Timing

- **20 documents**: ~30 seconds
- **100 documents**: ~2-3 minutes
- **1000 documents**: ~15-20 minutes

(Times vary based on API response time and pagination)

## Verifying Output

After running, check your corpus:

```bash
# Check files were created
ls -lh corpus/

# Count documents
wc -l corpus/opinions.jsonl

# View metadata
cat corpus/metadata.json | python -m json.tool

# View first document
head -n 1 corpus/opinions.jsonl | python -m json.tool
```

## Troubleshooting

### "Access denied" or 403 errors
- Make sure you're running locally, not in a cloud/container
- Verify your API token is correct
- Check you have an active CourtListener account

### "No documents were sampled"
- Check your internet connection
- Verify the API is up: https://www.courtlistener.com
- Try a different sampling strategy

### "Rate limit exceeded"
- Wait an hour (5,000 requests/hour limit)
- Reduce your `--count` value
- Check your API usage at: https://www.courtlistener.com/profile/

### Token in command history
If you used Option 3 and want to clear your token from shell history:

```bash
# Bash
history -d $(history | tail -n 2 | head -n 1 | awk '{print $1}')

# Or just close and reopen your terminal
```

## What The Code Does

The sampler:
1. Authenticates with CourtListener API
2. Queries for opinions based on strategy
3. Handles pagination automatically
4. Respects rate limits (0.8s between requests)
5. Saves results in JSON/JSONL format
6. Generates metadata about the corpus

## Sample Output Structure

Each document in `opinions.jsonl`:
```json
{
  "id": 12345,
  "cluster_id": 67890,
  "court": "scotus",
  "date_filed": "2023-06-15",
  "case_name": "Smith v. Jones",
  "text": "Full opinion text...",
  "url": "/opinion/12345/smith-v-jones/"
}
```

## Security Notes

- Never commit `config.json` (it's in `.gitignore`)
- Don't share your API token publicly
- Use environment variables for CI/CD
- Rotate tokens if accidentally exposed

## Next Steps

After successfully sampling:
1. Review the metadata.json to understand your corpus
2. Check the distribution of courts, dates, etc.
3. Process the corpus for your specific research needs
4. Consider which sampling strategy best fits your use case

## Getting Help

If you still have issues:
1. Check the main [README.md](README.md)
2. Review CourtListener docs: https://www.courtlistener.com/help/api/
3. Open an issue with error details
