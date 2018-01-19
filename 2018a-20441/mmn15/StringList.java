/**
 * A linked list of CharNode. Represents a string.
 */
public class StringList {

    private CharNode _head;

    /**
     * Creates a new, empty, StringList
     */
    public StringList()
    {
        _head = null;
    }

    /**
     * Creates a new StringList, starting with the given node
     * @param node The head of the list
     */
    public StringList(CharNode node) {
        if (node == null){
            _head = null;
        }
        else {
            _head = new CharNode(node.getData(), node.getValue(), null);

            for (CharNode ptr = node.getNext(), last = _head; ptr != null; ptr = ptr.getNext()) {
                last.setNext(new CharNode(ptr.getData(), ptr.getValue(), ptr.getNext()));
                last = last.getNext();
            }
        }
    }
}