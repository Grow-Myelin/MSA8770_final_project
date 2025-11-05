# Quick Start Guide

Get started sampling from CourtListener in 5 minutes!

## Step 1: Get Your API Token

1. Go to https://www.courtlistener.com/sign-in/register/
2. Create a free account
3. Visit https://www.courtlistener.com/api/rest-info/ to get your API token
4. Copy your token (it will look like: `abc123def456...`)

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Run Your First Sample

```bash
# Sample 100 documents (faster for testing)
python courtlistener_sampler.py --token YOUR_TOKEN_HERE --count 100

# Or for the full 1000 documents
python courtlistener_sampler.py --token YOUR_TOKEN_HERE --count 1000
```

## Step 4: Check Your Results

Your corpus will be saved in the `corpus/` directory:

```bash
ls corpus/
# You'll see:
#   - opinions.json      (single JSON array)
#   - opinions.jsonl     (one JSON per line)
#   - metadata.json      (corpus statistics)
```

## Step 5: Explore Your Data

### View metadata:
```bash
cat corpus/metadata.json
```

### Count documents:
```bash
wc -l corpus/opinions.jsonl
```

### View first document:
```bash
head -n 1 corpus/opinions.jsonl | python -m json.tool
```

## Advanced: Try Different Strategies

### Temporal Diversity (stratified across 20 years):
```bash
python courtlistener_sampler.py --token YOUR_TOKEN --strategy stratified --years 20
```

### Jurisdictional Diversity (multiple court types):
```bash
python courtlistener_sampler.py --token YOUR_TOKEN --strategy diverse
```

## Using as a Python Library

```python
from courtlistener_sampler import CourtListenerSampler

sampler = CourtListenerSampler(api_token="YOUR_TOKEN", target_count=100)
opinions = sampler.sequential_sample()
sampler.save_corpus(opinions, output_dir="my_corpus")
```

See `example_usage.py` for more examples!

## Troubleshooting

**"Authentication failed"**
- Double-check your token is correct
- Make sure there are no extra spaces or quotes

**"Rate limit exceeded"**
- Wait an hour for the limit to reset
- You get 5,000 requests per hour

**"No documents found"**
- Check your internet connection
- Try visiting https://www.courtlistener.com in your browser

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for programmatic usage
- Explore different sampling strategies
- Process your corpus for your specific needs

## Tips

1. **Start small**: Test with `--count 100` before running larger samples
2. **Use environment variables**: Store your token as `export COURTLISTENER_TOKEN=your_token`
3. **Choose the right strategy**:
   - `sequential`: Fastest, least diverse
   - `stratified`: Good for historical analysis
   - `diverse`: Good for multi-jurisdiction analysis
4. **Be patient**: 1000 documents takes about 10-15 minutes
5. **Check metadata**: Always review `metadata.json` to understand your corpus

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Visit https://www.courtlistener.com/help/api/ for API documentation
- Open an issue if you find bugs or have questions

---

Happy sampling! 🏛️
