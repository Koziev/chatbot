using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class FootPrintTrieNode
{
    public static string END_TOKEN = "<end>";
    public string token;
    public List<FootPrintTrieNode> next_nodes = new List<FootPrintTrieNode>();

    public FootPrintTrieNode(string token) { this.token = token; }


    public override string ToString()
    {
        return token;
    }

    public void Build(IReadOnlyList<string> tokens, int cur_index)
    {
        string next_token = tokens[cur_index];

        bool found = false;
        foreach (var next_node in next_nodes)
        {
            if (next_node.token == next_token)
            {
                if (cur_index < tokens.Count - 1)
                {
                    next_node.Build(tokens, cur_index + 1);
                }
                found = true;
            }
        }

        if (!found)
        {
            FootPrintTrieNode new_next_node = new FootPrintTrieNode(next_token);
            next_nodes.Add(new_next_node);
            if (cur_index < tokens.Count - 1)
            {
                new_next_node.Build(tokens, cur_index + 1);
            }
        }
    }

    public bool FindMatching(IReadOnlyList<FootPrintToken> tokens, int cur_index)
    {
        if (cur_index == -1 || tokens[cur_index].Match(token))
        {
            if (tokens.Count == cur_index + 1)
            {
                return token==END_TOKEN;
            }

            foreach (var next_node in next_nodes)
            {
                if (next_node.FindMatching(tokens, cur_index + 1))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
