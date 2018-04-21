using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class FootPrintTrie
{
    private FootPrintTrieNode root;

    public FootPrintTrie()
    {
        root = new FootPrintTrieNode(string.Empty);
    }

    // В дерево добавляется очередной пример валидной конструкции
    public void AddSequence(IReadOnlyList<string> tokens)
    {
        root.Build(tokens, 0);
    }

    // Проверяем, является ли цепочка токенов предложения синтаксически валидной.
    public bool Verify(IReadOnlyList<FootPrintToken> tokens)
    {
        return root.FindMatching(tokens, -1);
    }
}
