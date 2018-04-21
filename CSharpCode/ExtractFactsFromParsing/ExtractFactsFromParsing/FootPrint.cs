using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;


public class FootPrint
{
    private List<FootPrintToken> tokens;

    public FootPrint(SolarixGrammarEngineNET.GrammarEngine2 gren, List<SolarixGrammarEngineNET.SyntaxTreeNode> terms)
    {
        tokens = new List<FootPrintToken>();
        foreach (var term in terms)
        {
            tokens.Add(new FootPrintToken(gren, term));
        }

        tokens.Add(new FootPrintToken(FootPrintTrieNode.END_TOKEN));
    }

    public bool Match(string tags)
    {
        string[] groups = tags.Split(" ".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
        if (groups.Length != tokens.Count)
        {
            return false;
        }

        for (int i = 0; i < groups.Length; ++i)
        {
            if (!tokens[i].Match(groups[i]))
            {
                return false;
            }
        }

        return true;
    }


    public bool Match(FootPrintTrie trie)
    {
        return trie.Verify(tokens);
    }

    public FootPrintToken this[int index] { get { return tokens[index]; } }

    public override string ToString()
    {
        return string.Join(" ", tokens.Select(z => z.ToString()));
    }
}
