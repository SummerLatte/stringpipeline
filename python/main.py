import json
import sys
from json import JSONDecodeError

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QInputDialog
from PyQt5.uic import loadUi


class TextEditorApp(QMainWindow):

    def __init__(self):
        super().__init__()

        try:
            with open('scriptConfig.json', 'r') as f:
                self.scriptConfig = json.load(f)
        except JSONDecodeError:
            self.scriptConfig = {}

        loadUi('mainwindow.ui', self)

        # connect buttons to slots
        self.applyButton.clicked.connect(self.applyScript)
        self.saveButton.clicked.connect(self.saveScript)

        # Create a QStandardItemModel to hold the data
        self.model = QStandardItemModel()

        for scriptName in self.scriptConfig:
            # Add an item to the model
            item = QStandardItem(scriptName)
            self.model.appendRow(item)

        self.scriptListView.setModel(self.model)
        # Connect the clicked signal to a callback function
        self.scriptListView.clicked.connect(self.item_clicked)

    def applyScript(self):
        targetStr = self.targetTextEdit.toPlainText()
        scriptText = self.scriptTextEdit.toPlainText()
        print({scriptText})
        myLocal = {"targetStr": targetStr}
        exec(scriptText, globals(), myLocal)
        targetStr = myLocal["targetStr"]
        self.targetTextEdit.setPlainText(targetStr)

    def saveScript(self):
        scriptName, ok = QInputDialog.getText(self, "script name", "create script name:")
        if ok:
            self.scriptConfig[scriptName] = self.scriptTextEdit.toPlainText()

        with open('scriptConfig.json', 'w') as f:
            json.dump(self.scriptConfig, f)

        item = QStandardItem(scriptName)
        self.model.appendRow(item)

    def item_clicked(self, index):
        item = self.model.itemFromIndex(index)
        text = item.text()
        self.scriptTextEdit.setPlainText(self.scriptConfig[text])
        print(f'Item {text} was clicked!')


if __name__ == '__main__':
    "haha".upper()
    app = QApplication(sys.argv)
    editor = TextEditorApp()
    editor.show()
    sys.exit(app.exec_())
