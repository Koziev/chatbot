using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


class SNode
{
    public int index;
    public SToken word;
    public SNode parent;
    public List<string> edge_types = new List<string>();
    public List<SNode> children = new List<SNode>();

    public bool children_vector_ready = false;
    public bool parent_vector_ready = false;

    public override string ToString()
    {
        return word.ToString();
    }

    public List<SNode> FindEdge(string edge_type)
    {
        List<SNode> res = new List<SNode>();
        for (int i = 0; i < edge_types.Count; ++i)
            if (edge_types[i].Equals(edge_type, StringComparison.CurrentCultureIgnoreCase))
                res.Add(children[i]);

        return res;
    }

    public bool IsPartOfSpeech(params string[] part_of_speech)
    {
        foreach (var p in part_of_speech)
            if (word.part_of_speech.Equals(p, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }

    public bool IsPartOfSpeech(string part_of_speech)
    {
        if (word.part_of_speech.Equals(part_of_speech, StringComparison.CurrentCultureIgnoreCase))
            return true;

        return false;
    }

    public void FindNodesByClass(List<SNode> nodes, params string[] part_of_speech)
    {
        if (IsPartOfSpeech(part_of_speech))
            nodes.Add(this);

        foreach (SNode c in children)
            c.FindNodesByClass(nodes, part_of_speech);

        return;
    }

    public List<SNode> FindNodesByClass(string part_of_speech)
    {
        List<SNode> res = new List<SNode>();

        if (IsPartOfSpeech(part_of_speech))
            res.Add(this);

        foreach (SNode c in children)
            res.AddRange(c.FindNodesByClass(part_of_speech));

        return res;
    }

    public bool EqualsWord(params string[] probes)
    {
        foreach (string probe in probes)
            if (probe.Equals(word.word, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }


    public string ConvertTreeString()
    {
        System.Text.StringBuilder b = new StringBuilder();

        List<SNode> all = new List<SNode>();
        all.Add(this);
        all.AddRange(children);

        return string.Join(" ", all.OrderBy(z => z.word.index).Select(z => z.word.word).ToArray());
    }
}
