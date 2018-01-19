using System;
using System.Collections.Generic;
using System.Xml;

class SToken
{
    public int index;
    public string word;
    public string lemma;
    public string part_of_speech;
    public List<string> tags;

    public SToken(int _index, XmlNode n_token)
    {
        index = _index;
        word = n_token.SelectSingleNode("word").InnerText;
        lemma = n_token.SelectSingleNode("lemma").InnerText;

        if (n_token.SelectSingleNode("part_of_speech") != null)
            part_of_speech = n_token.SelectSingleNode("part_of_speech").InnerText;

        tags = new List<string>();
        if (n_token.SelectSingleNode("tags") != null)
        {
            tags.AddRange(n_token.SelectSingleNode("tags").InnerText.Split('|'));
        }
    }

    public override string ToString()
    {
        return word;
    }


    public bool ContainsTag(string tag)
    {
        foreach (string t in tags)
            if (t.Equals(tag, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }
}
