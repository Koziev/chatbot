using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class SyntaxChecker
{
    private FootPrintTrie trie;

    public SyntaxChecker() { }

    public void LoadTemplates(string path)
    {
        trie = new FootPrintTrie();
        using (System.IO.StreamReader rdr = new System.IO.StreamReader(path))
        {
            while (!rdr.EndOfStream)
            {
                string template = rdr.ReadLine();
                if (template == null) break;
                template = template.Trim();
                if (template.Length > 0)
                {
                    string[] parts = template.Split('#');
                    string sample = parts[0].Trim();
                    List<string> tokens = sample.Split(" ".ToCharArray(), StringSplitOptions.RemoveEmptyEntries).ToList();
                    tokens.Add(FootPrintTrieNode.END_TOKEN);
                    trie.AddSequence(tokens);
                }
            }
        }
    }

    public bool IsEmpty() => trie == null;

    public bool IsGoodSyntax(FootPrint footprint)
    {
        /*
                foreach (var good_footprint in good_footprints)
                {
                    if (footprint.Match(good_footprint))
                    {
                        return true;
                    }
                }

                return false;
        */

        return footprint.Match(trie); ;
    }

}
