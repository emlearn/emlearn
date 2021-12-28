
window.addEventListener('load', (_event) => {
    var menu = document.querySelector(".wy-menu ul li:first-child")
    recurse(menu)
});

/**
 * Given a Node, it recursively goes through every child and checks if the child is expandable, it
 * expands it unless it is already expanded.
 * 
 * @param {Node} node 
 */
function recurse(node) {
    if (is_expandable(node) && !is_expanded(node)) {
        node.classList.add("current")
    }

    // By default, children are not arrays, so we need to convert them
    children = Array.prototype.slice.call(node.children)

    children.forEach(recurse)
}

/**
 * Returns whether or not the given node is an expandable list.
 * 
 * @param {Node} node 
 * @returns {boolean} true if the node is a toctree that can be expanded, false otherwise.
 */
function is_expandable(node) {
    return node.className.includes("toctree-l")
}

/**
 * Returns whether or not the given expandable node is already expanded.
 * Nodes are considered expandaded if they are 'current'ly selected, so we take advantage of this.
 * 
 * @param {Node} node 
 * @returns {boolean} true if the node is already expanded, false otherwise.
 */
function is_expanded(node) {
    return node.classList.contains("current")
}
