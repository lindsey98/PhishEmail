/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global document, Office, fetch, console */

const SERVER = "http://localhost:5000";

// Remembered from the last analysis so "Trust this sender" can whitelist it.
let currentSender = "";
let currentIdentity = "";

const VERDICT_ICONS: { [k: string]: string } = {
  danger:
    '<svg viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2V9h2v5z"/></svg>',
  caution:
    '<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>',
  safe:
    '<svg viewBox="0 0 24 24"><path d="M12 2 4 5v6c0 5 3.4 8.4 8 10 4.6-1.6 8-5 8-10V5l-8-3zm-1 14-4-4 1.4-1.4L11 13.2l4.6-4.6L17 10l-6 6z"/></svg>',
  neutral:
    '<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-2h2v2zm1.1-7.3-.9.9c-.5.5-.7.9-.7 1.9h-2v-.5c0-.7.3-1.4.9-2l1.2-1.2a1.5 1.5 0 1 0-2.6-1H8a3.5 3.5 0 1 1 6.1 2.9z"/></svg>',
};

function setVerdict(state: string, label: string) {
  const banner = document.getElementById("banner");
  banner.className = "verdict " + state;
  document.getElementById("verdict-label").textContent = label;
  document.getElementById("verdict-icon").innerHTML = VERDICT_ICONS[state] || "";
}

// Show the verdict as a banner on the ORIGINAL email (top of the reading pane).
// Read mode can't highlight the body, but it can post a notification message.
// Phishing uses ErrorMessage (no manifest icon needed); others use an
// InformationalMessage (its `icon` must be a manifest icon resource — if absent
// the info banner is simply skipped, the phishing banner always shows).
function showNotification(state: string, message: string) {
  try {
    const item = Office.context.mailbox.item;
    if (!item || !item.notificationMessages) return;
    const key = "pimrefVerdict";
    const details: any = {
      type:
        state === "danger"
          ? Office.MailboxEnums.ItemNotificationMessageType.ErrorMessage
          : Office.MailboxEnums.ItemNotificationMessageType.InformationalMessage,
      message: String(message).substring(0, 150),
    };
    if (details.type === Office.MailboxEnums.ItemNotificationMessageType.InformationalMessage) {
      details.icon = "icon16";
      details.persistent = false;
    }
    // remove-then-add so re-analysis / trusting updates the same banner
    item.notificationMessages.removeAsync(key, () => {
      item.notificationMessages.addAsync(key, details, () => {});
    });
  } catch (error) {
    console.error("Notification error:", error);
  }
}

Office.onReady((info) => {
  if (info.host === Office.HostType.Outlook) {
    document.getElementById("sideload-msg").style.display = "none";
    document.getElementById("app-body").style.display = "flex";
    document.getElementById("run").onclick = run;
    document.getElementById("whitelist-btn").onclick = whitelistSender;
  }
});

function showExplanation(body, sender, recepient, subject) {
  const explanationButton = document.getElementById("detailed-explanation");
  const backButton = document.getElementById("return");
  const explanationBody = document.getElementById("html-body");
  explanationBody.innerHTML = body;
  explanationButton.onclick = () => {
    document.getElementById("tab-content").style.transform = "translateX(-100%)";
    setTimeout(function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 400);
  };
  backButton.onclick = () => {
    document.getElementById("tab-content").style.transform = "translateX(0%)";
    setTimeout(function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, 400);
  };

  document.getElementById("sender-info").innerHTML = sender;
  document.getElementById("recepient-info").innerHTML = recepient;
  document.getElementById("subject").innerHTML = subject;
}

function getActionExplanation(actions) {
  if (actions.length) {
    return `There are detected actions such as <em>"${actions[0]}"</em>, representing potentially malicious follow-up actions.`;
  } else {
    return `There are no detected actions in the email.`;
  }
}

// POST the last-analyzed (sender, claimed-identity) to the whitelist so this
// sender is no longer flagged for that identity.
async function whitelistSender() {
  const btn = document.getElementById("whitelist-btn") as HTMLButtonElement;
  if (!currentSender) return;
  btn.disabled = true;
  try {
    const resp = await fetch(SERVER + "/whitelist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sender: currentSender, matchedIdentity: currentIdentity }),
    });
    const data = await resp.json();
    if (data.status === "success") {
      setVerdict("safe", "Trusted sender");
      showNotification("safe", "PiMRef: trusted sender");
      btn.style.display = "none";
    } else {
      btn.disabled = false;
    }
  } catch (error) {
    console.error("Whitelist error:", error);
    btn.disabled = false;
  }
}

function handleResponse(isInconsistent, matchedIdentity, identities, actions, sender, whitelisted) {
  currentSender = sender;
  currentIdentity = matchedIdentity;

  // ---- verdict banner (task pane) + notification (original email) ----
  let state: string, label: string, notif: string;
  if (whitelisted) {
    state = "safe"; label = "Trusted sender";
    notif = `PiMRef: trusted sender${matchedIdentity ? ` (${matchedIdentity})` : ""}`;
  } else if (isInconsistent) {
    state = "danger"; label = "Phishing Alert";
    notif = `PiMRef: possible impersonation — claims "${matchedIdentity}", sent from ${sender}`;
  } else if (matchedIdentity != "No Matched Brand" && matchedIdentity != "No Prediction") {
    state = "safe"; label = "Likely Safe";
    notif = `PiMRef: sender identity verified (${matchedIdentity})`;
  } else {
    state = "neutral"; label = "No Match";
    notif = "PiMRef: could not verify the sender's identity";
  }
  setVerdict(state, label);
  showNotification(state, notif);

  const wrapper = document.getElementById("results-wrapper");
  const representedBrand = document.getElementById("represented-brand");
  const senderDomain = document.getElementById("sender-domain");
  const identityExplanation = document.getElementById("identity-explanation");
  const actionSection = document.getElementById("action");
  const actionExplanation = document.getElementById("action-explanation");
  const recommendation = document.getElementById("recommendation-explanation");
  senderDomain.textContent = sender;
  actionSection.style.display = "";

  // handle phishing
  actionExplanation.innerHTML = getActionExplanation(actions);
  if (isInconsistent) {
    if (matchedIdentity.startsWith("Internal")) {
      let role = matchedIdentity.split("Internal:")[1];
      representedBrand.textContent = `Internal Role: ${role}`;
      identityExplanation.innerHTML =
        "The email claims to be from an internal role but uses an email address inconsistent with the recepient's domain, representing <strong>identity inconsistency.</strong>";
    } else {
      representedBrand.textContent = matchedIdentity;
      identityExplanation.innerHTML =
        "There is a <strong>detected mismatch</strong> between the represented brand & sender domain, representing <strong>identity inconsistency.</strong>";
    }
    recommendation.innerHTML =
      "Please exercise extreme caution and <strong>avoid opening</strong> any links or attachments in this email.";
    // handle no matches
  } else if (matchedIdentity == "No Matched Brand") {
    representedBrand.textContent = identities[0];
    identityExplanation.innerHTML = `The email claims to be from <strong>${identities[0]}</strong>, but this does not match any known identity in our database.`;
    recommendation.innerHTML =
      "As PiMRef is unable to verify the legitimacy of this email, <strong>exercise caution</strong> with any links or attachments included in this email.";
  } else if (matchedIdentity == "No Prediction") {
    representedBrand.textContent = "Unidentifiable";
    identityExplanation.innerHTML = "Unfortunately, PiMRef is unable to identify any identities in this email.";
    recommendation.innerHTML =
      "As PiMRef is unable to verify the legitimacy of this email, <strong>exercise caution</strong> with any links or attachments included in this email.";
    // handle benign: Consistent
  } else {
    if (matchedIdentity.startsWith("Consistent")) {
      actionSection.style.display = "none";
      if (matchedIdentity.startsWith("Consistent:")) {
        // internal consistent
        let role = matchedIdentity.split("Consistent:")[1];
        representedBrand.textContent = `Internal Role: ${role}`;
        identityExplanation.innerHTML =
          "This email claims to be from an internal role, and there is consistency between the sender and recepient's domain.";
      } else {
        representedBrand.textContent = matchedIdentity;
        identityExplanation.innerHTML = `The email claims to be from ${matchedIdentity} and uses an email address which is consistent with the identity's domain.`;
      }
      recommendation.innerHTML =
        "PiMRef has not detected any phishing characteristics in this email. Thus, you are advised to exercise minimal caution when opening any links or attachments included in this email.";
      // handle benign: no action
    } else {
      if (matchedIdentity.startsWith("Internal")) {
        let role = matchedIdentity.split("Internal:")[1];
        representedBrand.textContent = `Internal Role: ${role}`;
        identityExplanation.innerHTML =
          "The email claims to be from an internal role but uses an email address inconsistent with the recepient's domain, representing <strong>identity inconsistency.</strong>";
      } else {
        representedBrand.textContent = matchedIdentity;
        identityExplanation.innerHTML =
          "There is a <strong>detected mismatch</strong> between the represented brand & sender domain, representing <strong>identity inconsistency.</strong>";
      }
      recommendation.innerHTML =
        "PiMRef has detected identity inconsistency in this email, but did not spot any signs of follow-up actions. However, do still exercise caution when <strong>opening any</strong> links or attachments in this email.";
    }
  }

  // ---- "Trust this sender" button visibility ----
  const wl = document.getElementById("whitelist-btn") as HTMLButtonElement;
  const hasIdentity = matchedIdentity != "No Prediction";
  if (whitelisted || !(isInconsistent || hasIdentity)) {
    wl.style.display = "none";
  } else {
    wl.style.display = "";
    wl.disabled = false;
  }

  wrapper.style.visibility = "visible";
  wrapper.style.opacity = "1";
}

// Package function into promise
function getAttachmentContentAsyncPromise(itemId, itemName) {
  return new Promise((resolve, reject) => {
    Office.context.mailbox.item.getAttachmentContentAsync(itemId, (asyncResult) => {
      if (asyncResult.status === Office.AsyncResultStatus.Succeeded) {
        resolve({ name: itemName, content: asyncResult.value.content });
      } else {
        reject(asyncResult.error);
      }
    });
  });
}

export async function run() {
  document.getElementById("run").style.display = "none";
  document.getElementById("loading").style.visibility = "visible";
  document.getElementById("loading").style.opacity = "1";

  const item = Office.context.mailbox.item;
  // Get all image attachments from email. Note that this will not work for encrypted emails
  let attachments = item.attachments;
  let attachmentData = [];
  let promises = [];

  if (attachments && item.attachments.length > 0) {
    if (!attachments[0].id) {
      console.error("Email is Encrypted and Attachments cannot be accessed!");
    } else {
      for (let i = 0; i < attachments.length; i++) {
        let extension = attachments[i].name.split(".").pop();
        if (extension == "png" || extension == "jpg") {
          promises.push(getAttachmentContentAsyncPromise(attachments[i].id, attachments[i].name));
        }
      }
    }
  }

  // consolidate all the results, then proceed with the next step
  await Promise.all(promises).then((results) => {
    attachmentData = results;
  });

  // send request to our server
  item.body.getAsync(Office.CoercionType.Html, function (result) {
    if (result.status === Office.AsyncResultStatus.Succeeded) {
      let email_data = {
        subject: item.subject,
        sender: item.sender,
        to: item.to,
        bcc: item.bcc,
        cc: item.cc,
        body: result.value,
        attachments: attachmentData,
      };

      fetch(SERVER + "/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(email_data),
      })
        .then((response) => {
          return response.json();
        })
        .then((responseData) => {
          document.getElementById("loading").style.display = "none";
          if (responseData.status == "error") throw new Error(responseData.message);
          handleResponse(
            responseData.isInconsistent,
            responseData.matchedIdentity,
            responseData.identities,
            responseData.actions,
            item.sender.emailAddress,
            responseData.whitelisted
          );
          showExplanation(
            responseData.labelledHTML,
            responseData.labelledSender,
            responseData.labelledRecipient,
            responseData.labelledSubject
          );
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } else {
      console.error("Error retrieving body HTML:", result.error.message);
    }
  });
}
